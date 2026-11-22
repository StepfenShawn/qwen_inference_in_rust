//! The tensor core: one representation for weights and activations alike.

pub mod dtype;
pub mod shape;

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::fmt;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;
use std::slice;

use dtype::DType;
use shape::Shape;

/// Byte alignment of every buffer this module allocates.
///
/// One cache line, so vector loads stay on the aligned path and a future
/// direct-IO or device-upload route does not have to re-allocate.
pub const TENSOR_ALIGN: usize = 64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// A row does not cover a whole number of quantized blocks.
    NotBlockAligned {
        cols: usize,
        block_elements: usize,
    },
    /// The byte count derived from the shape is not the byte count supplied.
    ByteLenMismatch {
        expected: usize,
        got: usize,
    },
    /// The requested row is outside the tensor.
    RowOutOfBounds {
        row: usize,
        rows: usize,
    },
    /// A reshape kept the element count but moved the quantized block
    /// boundaries, so the packed payload would be read differently.
    IncompatibleReshape,
    /// The shape does not fit in the address space.
    TooLarge,
    /// A `f32` view was asked of a tensor that does not store `f32`.
    NotF32(DType),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBlockAligned {
                cols,
                block_elements,
            } => write!(
                f,
                "{cols} cols do not cover whole {block_elements}-element blocks"
            ),
            Self::ByteLenMismatch { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            }
            Self::RowOutOfBounds { row, rows } => write!(f, "row {row} is outside [0, {rows})"),
            Self::IncompatibleReshape => write!(f, "reshape moves quantized block boundaries"),
            Self::TooLarge => write!(f, "tensor exceeds the address space"),
            Self::NotF32(dtype) => write!(f, "not an f32 tensor: {dtype:?}"),
        }
    }
}

impl std::error::Error for TensorError {}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Bytes a tensor of this shape and format occupies.
fn nbytes_of(shape: Shape, dtype: DType) -> Result<usize> {
    let cols = shape.cols();
    let row = dtype
        .row_bytes(cols)
        .ok_or(TensorError::NotBlockAligned {
            cols,
            block_elements: dtype.block_elements() as usize,
        })?;
    shape.rows().checked_mul(row).ok_or(TensorError::TooLarge)
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// Read-only window onto a tensor that lives somewhere else: a GGUF mapping, a
/// [`TensorBuf`], or a caller's slice.
///
/// `Copy`, and only as large as the shape plus a slice — it is passed by value
/// and carries no ownership.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    shape: Shape,
    dtype: DType,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Bind a byte slice to a shape, checking that the slice is exactly as long
    /// as the shape and format require.
    pub fn new(shape: Shape, dtype: DType, data: &'a [u8]) -> Result<Self> {
        let expected = nbytes_of(shape, dtype)?;
        if data.len() != expected {
            return Err(TensorError::ByteLenMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, dtype, data })
    }

    /// View over `f32` storage without copying.
    pub fn from_f32(shape: Shape, data: &'a [f32]) -> Result<Self> {
        let len = data.len() * size_of::<f32>();
        // SAFETY: `f32` is plain data with no padding, and viewing it as bytes
        // (alignment 1) cannot violate any invariant of the source slice.
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), len) };
        Self::new(shape, DType::F32, bytes)
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
        self.data.len()
    }

    /// Bytes of one row, including the block padding of quantized formats.
    pub fn row_bytes(&self) -> usize {
        self.data.len() / self.rows()
    }

    /// Raw bytes of row `row`.
    pub fn row(&self, row: usize) -> Result<&'a [u8]> {
        let rows = self.rows();
        if row >= rows {
            return Err(TensorError::RowOutOfBounds { row, rows });
        }
        let width = self.row_bytes();
        Ok(&self.data[row * width..(row + 1) * width])
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// The tensor as `f32` values, when it stores plain `f32`.
    ///
    /// Quantized tensors return `None` rather than dequantizing: decoding is a
    /// backend decision with a real cost, never something a view does silently.
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let ptr = self.data.as_ptr();
        if ptr.align_offset(align_of::<f32>()) != 0 {
            return None;
        }
        // SAFETY: `F32` means the length is a whole number of 4-byte elements,
        // the pointer is checked to be 4-byte aligned above, and every bit
        // pattern is a valid `f32`.
        Some(unsafe { slice::from_raw_parts(ptr.cast::<f32>(), self.data.len() / 4) })
    }

    /// Reinterpret the same bytes under a different shape.
    ///
    /// Only metadata moves, and both the element count and the quantized block
    /// boundaries have to survive — which is why a quantized tensor can only be
    /// re-cut across its outer axes.
    pub fn reshape(self, shape: Shape) -> Result<Self> {
        if shape.elems() != self.shape.elems() {
            return Err(TensorError::IncompatibleReshape);
        }
        if nbytes_of(shape, self.dtype)? != self.data.len() {
            return Err(TensorError::IncompatibleReshape);
        }
        Ok(Self { shape, ..self })
    }
}

impl fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorView({:?}, {:?})", self.shape, self.dtype)
    }
}

/// Mutable counterpart of [`TensorView`].
///
/// Not `Copy`: one writable view at a time is the whole point.
pub struct TensorViewMut<'a> {
    shape: Shape,
    dtype: DType,
    data: &'a mut [u8],
}

impl<'a> TensorViewMut<'a> {
    pub fn new(shape: Shape, dtype: DType, data: &'a mut [u8]) -> Result<Self> {
        let expected = nbytes_of(shape, dtype)?;
        if data.len() != expected {
            return Err(TensorError::ByteLenMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, dtype, data })
    }

    pub fn from_f32(shape: Shape, data: &'a mut [f32]) -> Result<Self> {
        let len = data.len() * size_of::<f32>();
        // SAFETY: as in `TensorView::from_f32`; the exclusive borrow moves into
        // the returned view, so no aliasing `f32` slice can exist beside it.
        let bytes = unsafe { slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), len) };
        Self::new(shape, DType::F32, bytes)
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
        self.data.len()
    }

    pub fn row_bytes(&self) -> usize {
        self.data.len() / self.rows()
    }

    pub fn row(&self, row: usize) -> Result<&[u8]> {
        let rows = self.rows();
        if row >= rows {
            return Err(TensorError::RowOutOfBounds { row, rows });
        }
        let width = self.row_bytes();
        Ok(&self.data[row * width..(row + 1) * width])
    }

    pub fn row_mut(&mut self, row: usize) -> Result<&mut [u8]> {
        let rows = self.rows();
        if row >= rows {
            return Err(TensorError::RowOutOfBounds { row, rows });
        }
        let width = self.row_bytes();
        Ok(&mut self.data[row * width..(row + 1) * width])
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let ptr = self.data.as_ptr();
        if ptr.align_offset(align_of::<f32>()) != 0 {
            return None;
        }
        // SAFETY: see `TensorView::as_f32`; the borrow is tied to `&self`.
        Some(unsafe { slice::from_raw_parts(ptr.cast::<f32>(), self.data.len() / 4) })
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let ptr = self.data.as_mut_ptr();
        let len = self.data.len() / 4;
        if ptr.align_offset(align_of::<f32>()) != 0 {
            return None;
        }
        // SAFETY: unique borrow of a 4-byte-aligned region of whole `f32`s.
        Some(unsafe { slice::from_raw_parts_mut(ptr.cast::<f32>(), len) })
    }
}

impl fmt::Debug for TensorViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorViewMut({:?}, {:?})", self.shape, self.dtype)
    }
}

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

/// An owned tensor: activation scratch, attention output, quantized activation
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HIDDEN: usize = 5120;

    #[test]
    fn shape_collapses_leading_axes_into_rows() {
        let flat = Shape::from2(4, 8);
        assert_eq!((flat.rows(), flat.cols(), flat.elems()), (4, 8, 32));

        // [heads, tokens, dim] addresses like a 2-D tensor with heads*tokens rows.
        let ranked = Shape::from3(4, 8, 16);
        assert_eq!(
            (ranked.rows(), ranked.cols(), ranked.elems()),
            (32, 16, 512)
        );

        let single = Shape::from1(HIDDEN);
        assert_eq!((single.rows(), single.cols()), (1, HIDDEN));
        assert_eq!(Shape::new(&[4, 8]).dims(), &[4u32, 8]);
    }

    #[test]
    fn row_bytes_follow_gguf_block_widths() {
        // 5120 = 20 blocks of 256, or 160 blocks of 32.
        assert_eq!(DType::F32.row_bytes(HIDDEN), Some(HIDDEN * 4));
        assert_eq!(DType::Q8_0.row_bytes(HIDDEN), Some(160 * 34));
        assert_eq!(DType::Q4K.row_bytes(HIDDEN), Some(20 * 144));
        assert_eq!(DType::Q6K.row_bytes(HIDDEN), Some(20 * 210));
        assert_eq!(DType::Iq1M.row_bytes(HIDDEN), Some(20 * 56));
        assert_eq!(DType::Iq4Xs.row_bytes(HIDDEN), Some(20 * 136));

        // A Q4_K row of 5120 weights is 2880 B, not 5120 B.
        assert_eq!(DType::Q4K.row_bytes(HIDDEN).unwrap(), 2880);
        // The fast path is unchanged: plain formats are their own block.
        assert_eq!(DType::F32.element_bytes(), Some(4));
        assert_eq!(DType::Q4K.element_bytes(), None);
    }

    #[test]
    fn row_bytes_reject_partial_blocks() {
        assert_eq!(DType::Q4K.row_bytes(100), None);
        assert_eq!(DType::Q8_0.row_bytes(100), None);
        assert_eq!(DType::F32.row_bytes(100), Some(400));
    }

    #[test]
    fn dtype_roundtrips_through_gguf_ids() {
        for id in [0u32, 1, 2, 8, 12, 14, 23, 29, 30] {
            let dtype = DType::from_u32(id).expect("known id");
            assert_eq!(dtype as u32, id);
        }
        assert!(DType::from_u32(4).is_none());
        assert!(DType::from_u32(99).is_none());
    }

    #[test]
    fn f32_view_roundtrips_and_slices_rows() {
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let view = TensorView::from_f32(Shape::from2(3, 8), &values).unwrap();

        assert_eq!(view.rows(), 3);
        assert_eq!(view.row_bytes(), 32);
        assert_eq!(view.as_f32().unwrap(), values.as_slice());

        let second = view.row(1).unwrap();
        assert_eq!(second.len(), 32);
        assert_eq!(f32::from_le_bytes(second[..4].try_into().unwrap()), 8.0);
        assert!(matches!(
            view.row(3),
            Err(TensorError::RowOutOfBounds { row: 3, rows: 3 })
        ));
    }

    #[test]
    fn new_rejects_short_buffers_and_bad_reshapes() {
        let bytes = [0u8; 64];
        assert!(matches!(
            TensorView::new(Shape::from2(3, 8), DType::F32, &bytes),
            Err(TensorError::ByteLenMismatch {
                expected: 96,
                got: 64
            })
        ));

        let view = TensorView::new(Shape::from2(2, 8), DType::F32, &bytes).unwrap();
        assert!(view.reshape(Shape::from2(1, 16)).is_ok());
        assert!(view.reshape(Shape::from2(3, 8)).is_err());

        // Re-cutting a quantized row would move its block boundaries.
        let packed = [0u8; 288];
        let quant = TensorView::new(Shape::from2(1, 512), DType::Q4K, &packed).unwrap();
        assert!(quant.reshape(Shape::from2(1, 256)).is_err());
        assert!(quant.as_f32().is_none());
    }

    #[test]
    fn buffer_rows_are_aligned_and_contiguous() {
        let mut buf = TensorBuf::zeros(Shape::from2(2, 8), DType::F32).unwrap();
        assert_eq!(buf.nbytes(), 64);
        assert_eq!(buf.as_ptr().addr() % TENSOR_ALIGN, 0);

        buf.row_mut(0).unwrap().as_f32_mut().unwrap().fill(1.0);
        buf.row_mut(1).unwrap().as_f32_mut().unwrap().fill(2.0);

        let flat = buf.as_f32().unwrap();
        assert_eq!(&flat[..8], &[1.0f32; 8]);
        assert_eq!(&flat[8..], &[2.0f32; 8]);
        assert!(buf.row(2).is_err());
    }

    #[test]
    fn quantized_buffer_allocates_whole_blocks() {
        let buf = TensorBuf::zeros(Shape::from3(2, 3, 512), DType::Q4K).unwrap();
        assert_eq!(buf.rows(), 6);
        assert_eq!(buf.nbytes(), 6 * 288);
        assert!(TensorBuf::zeros(Shape::from2(1, 100), DType::Q4K).is_err());
    }
}
