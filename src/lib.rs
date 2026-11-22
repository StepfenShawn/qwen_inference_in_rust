use ahash::HashSet;
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod gguf;
pub mod kernel;
pub mod model;
pub mod prompt;
pub mod server;
pub mod tensor;
pub mod tokenizer;

/// model-independent token state
pub struct TokenState {
    pub eos: u32,
    pub think_close: Option<u32>,
    pub control: HashSet<u32>,
    pub draft: u32,
}
