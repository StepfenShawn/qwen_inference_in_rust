#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetaType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }

    pub fn scalar_size(self) -> Option<u64> {
        Some(match self {
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
            _ => return None,
        })
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
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

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
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

    pub fn block_elements(self) -> u32 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Iq4Nl => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M => 256,
        }
    }

    pub fn block_bytes(self) -> u32 {
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
}

#[derive(Debug, Clone, Copy)]
pub struct GgufString<'a> {
    pub data: &'a [u8],
}

impl<'a> GgufString<'a> {
    pub fn as_str(&self) -> Option<&'a str> {
        std::str::from_utf8(self.data).ok()
    }

    pub fn eq_str(&self, other: &str) -> bool {
        self.data == other.as_bytes()
    }
}

#[derive(Debug, Clone)]
pub struct MetaEntry<'a> {
    pub key: GgufString<'a>,
    pub ty: MetaType,
    pub array_type: Option<MetaType>,
    pub count: u64,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct TensorEntry<'a> {
    pub name: GgufString<'a>,
    pub n_dims: u32,
    pub shape: [u64; 8],
    pub ty: GgmlType,
    pub offset: u64,
    pub nbytes: u64,
    pub data: &'a [u8],
    pub iq1_s_repack: Option<Box<[u8]>>,
}

impl MetaEntry<'_> {
    pub fn extract_string_array(&self) -> Vec<String> {
        let mut result = Vec::with_capacity(self.count as usize);
        let mut pos = 0 as usize;
        let data = self.data;
        for _ in 0..self.count {
            let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let s = std::str::from_utf8(&data[pos..pos + len]).unwrap();
            result.push(s.to_string());
            pos += len;
        }
        result
    }

    pub fn extract_i32_array(&self) -> Vec<i32> {
        let mut result = Vec::with_capacity(self.count as usize);
        let mut pos = 0 as usize;
        let data = self.data;
        for _ in 0..self.count {
            result.push(i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        result
    }
}
