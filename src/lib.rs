use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod gguf;
pub mod kernel;
pub mod model;
pub mod prompt;
pub mod tokenizer;
