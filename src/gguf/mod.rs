pub mod entry;

use memmap2::Mmap;
use std::{fs::File, path::Path};

pub type GgmlType = crate::gguf::entry::GgmlType;
pub type GgufString<'a> = crate::gguf::entry::GgufString<'a>;
pub type MetaEntry<'a> = crate::gguf::entry::MetaEntry<'a>;
pub type MetaType = crate::gguf::entry::MetaType;
pub type TensorEntry<'a> = crate::gguf::entry::TensorEntry<'a>;

pub const GGUF_MAX_DIMS: usize = 8;
const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    Invalid(&'static str),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::Invalid(msg) => write!(f, "invalid GGUF: {msg}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    at: usize,
    base: &'a [u8],
    failed: bool,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            at: 0,
            base: bytes,
            failed: false,
        }
    }

    fn take(&mut self, size: u64) -> Option<&'a [u8]> {
        if self.failed {
            return None;
        }
        let size = usize::try_from(size).ok()?;
        let end = self.at.checked_add(size)?;
        if end > self.base.len() {
            self.failed = true;
            return None;
        }
        let slice = &self.base[self.at..end];
        self.at = end;
        Some(slice)
    }

    fn u8(&mut self) -> u8 {
        self.take(1).map(|s| s[0]).unwrap_or(0)
    }
    fn u16(&mut self) -> u16 {
        self.take(2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]))
            .unwrap_or(0)
    }
    fn u32(&mut self) -> u32 {
        self.take(4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .unwrap_or(0)
    }
    fn u64(&mut self) -> u64 {
        self.take(8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
            .unwrap_or(0)
    }

    fn string(&mut self) -> Option<GgufString<'a>> {
        let length = self.u64();
        if self.failed {
            return None;
        }
        let data = self.take(length)?;
        Some(GgufString { data })
    }

    fn skip_meta_value(&mut self, ty: MetaType, count: u64) -> bool {
        if let Some(sz) = ty.scalar_size() {
            if count == 1 {
                return self.take(sz).is_some() && !self.failed;
            }
            if count > u64::MAX / sz {
                return false;
            }
            return self.take(count * sz).is_some();
        }
        if ty == MetaType::String {
            for _ in 0..count {
                if self.string().is_none() {
                    return false;
                }
            }
            return true;
        }
        false
    }
}

pub struct Gguf {
    _mmap: Mmap,
    pub version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    tensors: Vec<TensorEntry<'static>>,
    metadata: Vec<MetaEntry<'static>>,
}

impl Gguf {
    pub fn find_meta(&self, key: &str) -> Option<&MetaEntry<'static>> {
        self.metadata.iter().find(|m| m.key.eq_str(key))
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorEntry<'static>> {
        self.tensors.iter().find(|t| t.name.eq_str(name))
    }

    pub fn tensors(&self) -> &[TensorEntry<'static>] {
        &self.tensors
    }

    /// Mutable view used by optional kernel repacks; the mapped tensor data
    /// itself is never modified, only the derived views stored next to it.
    pub fn tensors_mut(&mut self) -> &mut [TensorEntry<'static>] {
        &mut self.tensors
    }

    pub fn metadata(&self) -> &[MetaEntry<'static>] {
        &self.metadata
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        let m = self.find_meta(key)?;
        if m.ty != MetaType::Uint32 || m.data.len() != 4 {
            return None;
        }
        Some(u32::from_le_bytes(m.data.try_into().unwrap()))
    }

    pub fn meta_string(&self, key: &str) -> Option<GgufString<'static>> {
        let m = self.find_meta(key)?;
        if m.ty != MetaType::String {
            return None;
        }
        Some(GgufString { data: m.data })
    }

    fn parse_metadata(
        &mut self,
        reader: &mut Reader<'static>,
        count: u64,
    ) -> Result<(), GgufError> {
        for _ in 0..count {
            let key = reader.string().ok_or(GgufError::Invalid("metadata key"))?;
            let ty_raw = reader.u32();
            if reader.failed {
                return Err(GgufError::Invalid("metadata type"));
            }
            let ty = MetaType::from_u32(ty_raw).ok_or(GgufError::Invalid("unknown meta type"))?;

            let entry = match ty {
                MetaType::String => {
                    let s = reader.string().ok_or(GgufError::Invalid("meta string"))?;
                    MetaEntry {
                        key,
                        ty,
                        array_type: None,
                        count: 1,
                        data: s.data,
                    }
                }
                MetaType::Array => {
                    let array_type_raw = reader.u32();
                    let array_type = MetaType::from_u32(array_type_raw)
                        .ok_or(GgufError::Invalid("unknown array type"))?;
                    if array_type == MetaType::Array {
                        return Err(GgufError::Invalid("nested arrays not allowed"));
                    }
                    let n = reader.u64();
                    let start = reader.at;
                    if !reader.skip_meta_value(array_type, n) {
                        return Err(GgufError::Invalid("array value"));
                    }
                    let data = &reader.base[start..reader.at];
                    MetaEntry {
                        key,
                        ty,
                        array_type: Some(array_type),
                        count: n,
                        data,
                    }
                }
                scalar => {
                    let sz = scalar
                        .scalar_size()
                        .ok_or(GgufError::Invalid("bad scalar size"))?;
                    let data = reader.take(sz).ok_or(GgufError::Invalid("scalar value"))?;
                    MetaEntry {
                        key,
                        ty,
                        array_type: None,
                        count: 1,
                        data,
                    }
                }
            };
            self.metadata.push(entry);
        }
        Ok(())
    }

    fn parse_tensors(&mut self, reader: &mut Reader<'static>, count: u64) -> Result<(), GgufError> {
        for _ in 0..count {
            let name = reader.string().ok_or(GgufError::Invalid("tensor name"))?;
            let n_dims = reader.u32();
            if reader.failed || n_dims == 0 || n_dims as usize > GGUF_MAX_DIMS {
                return Err(GgufError::Invalid("tensor n_dims"));
            }

            let mut shape = [0u64; GGUF_MAX_DIMS];
            let mut elements: u64 = 1;
            for d in 0..n_dims as usize {
                shape[d] = reader.u64();
                if shape[d] == 0 || elements > u64::MAX / shape[d] {
                    return Err(GgufError::Invalid("tensor shape overflow"));
                }
                elements *= shape[d];
            }
            let ty_raw = reader.u32();
            let offset = reader.u64();
            if reader.failed {
                return Err(GgufError::Invalid("tensor trailer"));
            }
            let ty = GgmlType::from_u32(ty_raw).ok_or(GgufError::Invalid("unknown tensor type"))?;

            let block_elements = ty.block_elements() as u64;
            let block_bytes = ty.block_bytes() as u64;
            if shape[0] % block_elements != 0
                || elements % block_elements != 0
                || elements / block_elements > u64::MAX / block_bytes
            {
                return Err(GgufError::Invalid("tensor bytes overflow"));
            }
            let nbytes = elements / block_elements * block_bytes;

            self.tensors.push(TensorEntry {
                name,
                n_dims,
                shape,
                ty,
                offset,
                nbytes,
                data: &[],
                iq1_s_repack: None,
            });
        }
        Ok(())
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let meta = file.metadata()?;
        if (meta.len() as usize) < 24 {
            return Err(GgufError::Invalid("file smaller than 24 bytes"));
        }

        let mmap = unsafe { Mmap::map(&file)? };
        let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

        let mut reader = Reader::new(bytes);
        let magic = reader.u32();
        let version = reader.u32();
        let tensor_count = reader.u64();
        let metadata_count = reader.u64();

        if magic != GGUF_MAGIC || !(2..=3).contains(&version) || tensor_count == 0 {
            return Err(GgufError::Invalid("unsupported or corrupt GGUF header"));
        }

        let mut gguf = Gguf {
            _mmap: mmap,
            version,
            alignment: 32,
            data_offset: 0,
            tensors: Vec::with_capacity(tensor_count as usize),
            metadata: Vec::with_capacity(metadata_count as usize),
        };

        gguf.parse_metadata(&mut reader, metadata_count)?;
        gguf.parse_tensors(&mut reader, tensor_count)?;

        if let Some(m) = gguf.find_meta("general.alignment") {
            if m.ty == MetaType::Uint32 && m.data.len() == 4 {
                let declared = u32::from_le_bytes(m.data.try_into().unwrap());
                if declared == 0 || (declared & (declared - 1)) != 0 {
                    return Err(GgufError::Invalid("invalid GGUF alignment"));
                }
                gguf.alignment = declared;
            }
        }

        let descriptor_end = reader.at as u64;
        let align = gguf.alignment as u64;
        let data_offset = descriptor_end
            .checked_add(align - 1)
            .ok_or(GgufError::Invalid("data_offset overflow"))?
            & !(align - 1);
        if data_offset > bytes.len() as u64 {
            return Err(GgufError::Invalid("tensor data begins past end of file"));
        }
        gguf.data_offset = data_offset;

        let mapping_size = bytes.len() as u64;
        for tensor in gguf.tensors.iter_mut() {
            let start = data_offset
                .checked_add(tensor.offset)
                .ok_or(GgufError::Invalid("tensor offset overflow"))?;
            let end = start
                .checked_add(tensor.nbytes)
                .ok_or(GgufError::Invalid("tensor nbytes overflow"))?;
            if end > mapping_size {
                return Err(GgufError::Invalid("tensor outside GGUF file"));
            }
            tensor.data = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr().add(start as usize),
                    tensor.nbytes as usize,
                )
            };
        }

        Ok(gguf)
    }
}
