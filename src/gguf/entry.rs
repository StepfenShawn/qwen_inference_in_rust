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
    pub ty: crate::tensor::dtype::DType,
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
