const CP2BYTE_MAX: usize = 512;

fn build_cp2byte() -> [i16; CP2BYTE_MAX] {
    let mut table = [-1i16; CP2BYTE_MAX];
    let mut extra: u32 = 0;
    for byte in 0u32..256 {
        let direct = (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte);
        let cp = if direct {
            byte
        } else {
            let cp = 256 + extra;
            extra += 1;
            cp
        };
        table[cp as usize] = byte as i16;
    }
    debug_assert_eq!(extra, 68, "expected 188 direct-mapped bytes");
    table
}

pub struct ByteDecoder {
    tokens: Vec<String>,
    added: Vec<bool>,
    cp2byte: [i16; CP2BYTE_MAX],
    pending: Vec<u8>,
}

impl ByteDecoder {
    pub fn new(tokens: Vec<String>, added: Vec<bool>) -> Self {
        Self {
            tokens,
            added,
            cp2byte: build_cp2byte(),
            pending: Vec::new(),
        }
    }
}

impl ByteDecoder {
    /// Drop any half-decoded bytes left over from a previous stream, so one
    /// decoder can serve many independent requests (a server, for instance).
    pub fn reset(&mut self) {
        self.pending.clear();
    }

    pub fn push(&mut self, id: u32) -> String {
        let idx = id as usize;
        if idx >= self.tokens.len() {
            return String::new();
        }

        if self.added.get(idx).copied().unwrap_or(false) {
            let mut out = self.take_pending();
            out.push_str(&self.tokens[idx]);
            return out;
        }

        let token = &self.tokens[idx];
        self.pending.reserve(token.len());
        for ch in token.chars() {
            match self.cp2byte.get(ch as usize).copied().unwrap_or(-1) {
                byte if byte >= 0 => self.pending.push(byte as u8),
                _ => {
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf).as_bytes().to_vec();
                    self.pending.extend_from_slice(&encoded);
                }
            }
        }
        self.drain_complete()
    }

    pub fn finish(&mut self) -> String {
        self.take_pending()
    }

    fn take_pending(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let bytes = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn drain_complete(&mut self) -> String {
        let bytes = std::mem::take(&mut self.pending);
        let mut out = String::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            match std::str::from_utf8(&bytes[i..]) {
                Ok(text) => {
                    out.push_str(text);
                    i = bytes.len();
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    if valid > 0 {
                        out.push_str(std::str::from_utf8(&bytes[i..i + valid]).unwrap());
                        i += valid;
                    }
                    match err.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            i += bad;
                        }
                        None => break,
                    }
                }
            }
        }
        self.pending.extend_from_slice(&bytes[i..]);
        out
    }
}
