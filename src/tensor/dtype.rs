#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    Iq1M = 29,
    Bf16 = 30,
}

impl DType {
    /// GGUF type id to format, or `None` when the id is unknown.
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            _ => return None,
        })
    }

    /// Elements stored in one block.
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Iq4Nl => 32,
            _ => 256,
        }
    }

    /// Bytes stored in one block.
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Iq4Nl => 18,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq4Xs => 136,
            Self::Iq1M => 56,
        }
    }

    /// True when one element is one stored value, so the buffer can be read as
    /// `f32`/`f16` directly.
    pub const fn is_plain(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }

    /// Bytes per element, for plain formats only.
    pub const fn element_bytes(self) -> Option<u32> {
        if self.is_plain() {
            Some(self.block_bytes())
        } else {
            None
        }
    }

    /// Bytes occupied by one row of `cols` elements, or `None` when the row does
    /// not cover a whole number of blocks.
    pub fn row_bytes(self, cols: usize) -> Option<usize> {
        let elements = self.block_elements() as usize;
        let bytes = self.block_bytes() as usize;
        if elements == 0 || bytes == 0 || cols == 0 || cols % elements != 0 {
            return None;
        }
        Some(cols / elements * bytes)
    }
}