use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::fmt;
use std::ptr::NonNull;
use std::slice;

use crate::tensor::TensorError;
use crate::tensor::dtype::DType;
use crate::tensor::nbytes_of;
use crate::tensor::shape::Shape;
use crate::tensor::views::{TensorView, TensorViewMut};

type Result<T> = std::result::Result<T, TensorError>;

/// Byte alignment of every buffer this module allocates.
///
/// One cache line, so vector loads stay on the aligned path and a future
/// direct-IO or device-upload route does not have to re-allocate.
pub const TENSOR_ALIGN: usize = 64;

// ---------------------------------------------------------------------------
// Owned storage
// ---------------------------------------------------------------------------

/// A zeroed, [`TENSOR_ALIGN`]-aligned heap block.
///
/// `Vec<u8>` promises byte alignment only, which would make an `f32` view — let
/// alone a vector load — undefined. This is the only place the module allocates
/// by hand, and it is deliberately fixed-size: growth is not something a
/// scratch buffer should do.
struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

impl AlignedBytes {
    fn zeroed(len: usize) -> Self {
        let cap = len.max(TENSOR_ALIGN).next_multiple_of(TENSOR_ALIGN);
        let layout = Layout::from_size_align(cap, TENSOR_ALIGN).expect("valid layout");
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            handle_alloc_error(layout)
        };
        Self { ptr, len, cap }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `len` initialised bytes for as long as `self`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and the borrow is exclusive.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // The layout must name the size `alloc_zeroed` used, hence `cap`.
        let layout = Layout::from_size_align(self.cap, TENSOR_ALIGN).expect("valid layout");
        // SAFETY: allocated by `zeroed` with this exact layout, freed once.
        unsafe { dealloc(self.ptr.as_ptr(), layout) }
    }
}

// SAFETY: the block is uniquely owned plain bytes, like `Vec<u8>`.
unsafe impl Send for AlignedBytes {}
// SAFETY: as above, and there is no interior mutability.
unsafe impl Sync for AlignedBytes {}

// An owned tensor: activation scratch, attention output, quantized activation
/// blocks.
///
/// Weights never appear here — they stay borrowed from the model file. This
/// type exists for the buffers the engine writes into, and always owns exactly
/// one allocation sized by its shape and format.
pub struct TensorBuf {
    shape: Shape,
    dtype: DType,
    data: AlignedBytes,
}

impl TensorBuf {
    /// Allocate `shape` for `dtype`, zero-filled.
    pub fn zeros(shape: Shape, dtype: DType) -> Result<Self> {
        let nbytes = nbytes_of(shape, dtype)?;
        Ok(Self {
            shape,
            dtype,
            data: AlignedBytes::zeroed(nbytes),
        })
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn nbytes(&self) -> usize {
        self.data.len
    }

    /// Start of the allocation; always [`TENSOR_ALIGN`]-aligned.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.ptr.as_ptr()
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: self.shape,
            dtype: self.dtype,
            data: self.data.as_slice(),
        }
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            shape: self.shape,
            dtype: self.dtype,
            data: self.data.as_mut_slice(),
        }
    }

    /// One row as its own tensor — the shape a kernel wants when it takes a
    /// single token.
    pub fn row(&self, row: usize) -> Result<TensorView<'_>> {
        let rows = self.rows();
        if row >= rows {
            return Err(TensorError::RowOutOfBounds { row, rows });
        }
        let width = self.nbytes() / rows;
        let shape = Shape::from2(1, self.cols());
        TensorView::new(
            shape,
            self.dtype,
            &self.data.as_slice()[row * width..(row + 1) * width],
        )
    }

    pub fn row_mut(&mut self, row: usize) -> Result<TensorViewMut<'_>> {
        let rows = self.rows();
        if row >= rows {
            return Err(TensorError::RowOutOfBounds { row, rows });
        }
        let width = self.nbytes() / rows;
        let shape = Shape::from2(1, self.cols());
        TensorViewMut::new(
            shape,
            self.dtype,
            &mut self.data.as_mut_slice()[row * width..(row + 1) * width],
        )
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let ptr = self.data.ptr.as_ptr();
        // SAFETY: the block is 64-byte aligned and `F32` makes the length a
        // whole number of 4-byte elements, so the reinterpretation is aligned
        // and in bounds; the borrow is tied to `&self`.
        Some(unsafe { slice::from_raw_parts(ptr.cast::<f32>(), self.data.len / 4) })
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let ptr = self.data.ptr.as_ptr();
        let len = self.data.len / 4;
        // SAFETY: as above, and the borrow is exclusive.
        Some(unsafe { slice::from_raw_parts_mut(ptr.cast::<f32>(), len) })
    }
}

impl fmt::Debug for TensorBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorBuf({:?}, {:?}, {} B)",
            self.shape,
            self.dtype,
            self.nbytes()
        )
    }
}
