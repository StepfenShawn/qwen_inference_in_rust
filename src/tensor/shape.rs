use std::fmt;

pub const MAX_DIMS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [u32; MAX_DIMS],
    ndim: u8,
}

impl Shape {
    /// Shape from axis lengths.
    ///
    /// # Panics
    /// On an empty shape, more than [`MAX_DIMS`] axes, or a zero-length axis.
    pub fn new(dims: &[usize]) -> Self {
        assert!(
            !dims.is_empty() && dims.len() <= MAX_DIMS,
            "shape must have 1..={MAX_DIMS} dims, got {}",
            dims.len()
        );
        let mut packed = [0u32; MAX_DIMS];
        for (axis, length) in dims.iter().enumerate() {
            assert!(*length > 0, "axis {axis} is empty");
            packed[axis] = *length as u32;
        }
        Self {
            dims: packed,
            ndim: dims.len() as u8,
        }
    }

    /// Row-major `[rows, cols]`.
    pub const fn from2(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "shape axes must be non-zero");
        Self {
            dims: [rows as u32, cols as u32, 0, 0],
            ndim: 2,
        }
    }

    /// A single row.
    pub const fn from1(cols: usize) -> Self {
        assert!(cols > 0, "shape axes must be non-zero");
        Self {
            dims: [cols as u32, 0, 0, 0],
            ndim: 1,
        }
    }

    pub const fn from3(rows: usize, middle: usize, cols: usize) -> Self {
        assert!(
            rows > 0 && middle > 0 && cols > 0,
            "shape axes must be non-zero"
        );
        Self {
            dims: [rows as u32, middle as u32, cols as u32, 0],
            ndim: 3,
        }
    }

    pub const fn ndim(&self) -> usize {
        self.ndim as usize
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.ndim as usize]
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.dims[axis] as usize
    }

    /// Innermost axis.
    pub fn cols(&self) -> usize {
        self.dims[self.ndim as usize - 1] as usize
    }

    /// Total elements.
    pub fn elems(&self) -> usize {
        self.dims().iter().map(|d| *d as usize).product()
    }

    /// Rows of `cols` elements: every leading axis collapsed.
    pub fn rows(&self) -> usize {
        self.elems() / self.cols()
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (axis, length) in self.dims().iter().enumerate() {
            if axis > 0 {
                f.write_str("x")?;
            }
            write!(f, "{length}")?;
        }
        Ok(())
    }
}
