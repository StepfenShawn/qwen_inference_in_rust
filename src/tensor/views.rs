use std::fmt;
use std::slice;

use crate::tensor::TensorError;
use crate::tensor::dtype::DType;
use crate::tensor::nbytes_of;
use crate::tensor::shape::Shape;

type Result<T> = std::result::Result<T, TensorError>;

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
    pub shape: Shape,
    pub dtype: DType,
    pub data: &'a [u8],
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
        if shape.elems() != self.shape.elems() || nbytes_of(shape, self.dtype)? != self.data.len() {
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
    pub shape: Shape,
    pub dtype: DType,
    pub data: &'a mut [u8],
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
