pub mod decoder;

use ahash::AHashMap;
use std::fmt::{self};
use tokenizers::{
    AddedToken, Tokenizer, decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BpeBuilder, pre_tokenizers::byte_level::ByteLevel,
};

use crate::{gguf::Gguf, tokenizer::decoder::ByteDecoder};

#[derive(Debug)]
pub enum TokenizerError {
    MissingField(String),
    TokenizerError(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingField(s) => write!(f, "MissingField: {}", s),
            TokenizerError::TokenizerError(s) => write!(f, "TokenizerError: {}", s),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub fn build_tokenizer_and_decoder_from_gguf(
    gguf: &Gguf,
) -> Result<(Tokenizer, ByteDecoder), TokenizerError> {
    build_with_decoder(gguf)
}

fn build_with_decoder(gguf: &Gguf) -> Result<(Tokenizer, ByteDecoder), TokenizerError> {
    let tokens = gguf
        .find_meta("tokenizer.ggml.tokens")
        .ok_or_else(|| {
            TokenizerError::MissingField("Missing `tokenizer.ggml.tokens` in .gguf file".into())
        })?
        .extract_string_array();

    let merges_raw = gguf
        .find_meta("tokenizer.ggml.merges")
        .ok_or_else(|| {
            TokenizerError::MissingField("Missing `tokenizer.ggml.merges` in .gguf file".into())
        })?
        .extract_string_array();

    let token_types = gguf
        .find_meta("tokenizer.ggml.token_type")
        .and_then(|e| Some(e.extract_i32_array()));

    let unk_token = find_special_token(&tokens, &token_types, "unknown")
        .unwrap_or_else(|| "<|endoftext|>".to_string());

    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|m| {
            let mut it = m.splitn(2, ' ');
            let a = it.next().unwrap_or("").to_string();
            let b = it.next().unwrap_or("").to_string();
            (a, b)
        })
        .collect();

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .unk_token(unk_token)
        .build()
        .map_err(|e| TokenizerError::TokenizerError(format!("failed to build BPE: {e}")))?;

    let mut tokenizer = Tokenizer::new(bpe);

    let pre_tok = ByteLevel::new(false, false, true);
    tokenizer.with_pre_tokenizer(Some(pre_tok));

    let decoder = ByteLevelDecoder::new(false, false, true);
    tokenizer.with_decoder(Some(decoder));

    if let Some(types) = &token_types {
        register_special_tokens(&mut tokenizer, &tokens, types)?;
    }

    let added: Vec<bool> = match &token_types {
        Some(types) => (0..tokens.len())
            .map(|i| matches!(types.get(i), Some(&t) if t == 3 || t == 4))
            .collect(),
        None => vec![false; tokens.len()],
    };

    let decoder = ByteDecoder::new(tokens, added);

    Ok((tokenizer, decoder))
}

pub fn get_all_control_token_ids(gguf: &Gguf) -> Vec<u32> {
    gguf.find_meta("tokenizer.ggml.token_type")
        .and_then(|entry| Some(entry.extract_i32_array()))
        .map(|types| {
            types
                .iter()
                .enumerate()
                .filter(|&(_, &token_type)| token_type == 3)
                .map(|(id, _)| id as u32)
                .collect()
        })
        .unwrap_or_default()
}

/// 根据 token_type 查找特殊 token
/// GGUF token_type 枚举: 1=Normal, 2=Unknown, 3=Control, 4=UserDefined, 5=Unused, 6=Byte
fn find_special_token(
    tokens: &[String],
    token_types: &Option<Vec<i32>>,
    kind: &str,
) -> Option<String> {
    let types = token_types.as_ref()?;
    let target_type = match kind {
        "unknown" => 2,
        "control" => 3,
        _ => return None,
    };
    for (i, &t) in types.iter().enumerate() {
        if t == target_type {
            return Some(tokens[i].clone());
        }
    }
    None
}

fn register_special_tokens(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    token_types: &[i32],
) -> Result<(), TokenizerError> {
    let added: Vec<AddedToken> = token_types
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == 3 || t == 4) // Control or UserDefined
        .map(|(i, _)| AddedToken::from(tokens[i].as_str(), true))
        .collect();

    if !added.is_empty() {
        tokenizer
            .add_special_tokens(added)
            .map_err(|e| TokenizerError::TokenizerError(format!("注册特殊 token 失败: {e}")))?;
    }
    Ok(())
}
