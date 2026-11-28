//! The tensor core: one representation for weights and activations alike.

pub mod buffers;
pub mod dtype;
pub mod shape;
pub mod views;

use std::fmt;

use dtype::DType;
use shape::Shape;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// A row does not cover a whole number of quantized blocks.
    NotBlockAligned { cols: usize, block_elements: usize },
    /// The byte count derived from the shape is not the byte count supplied.
    ByteLenMismatch { expected: usize, got: usize },
    /// The requested row is outside the tensor.
    RowOutOfBounds { row: usize, rows: usize },
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

/// Bytes a tensor of this shape and format occupies.
pub fn nbytes_of(shape: Shape, dtype: DType) -> std::result::Result<usize, TensorError> {
    let cols = shape.cols();
    let row = dtype.row_bytes(cols).ok_or(TensorError::NotBlockAligned {
        cols,
        block_elements: dtype.block_elements() as usize,
    })?;
    shape.rows().checked_mul(row).ok_or(TensorError::TooLarge)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::buffers::TENSOR_ALIGN;

    use buffers::TensorBuf;
    use views::TensorView;

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
