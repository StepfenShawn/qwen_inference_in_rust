pub mod web;

use crate::TokenState;
use crate::model::sampler::Sampler;
use crate::model::{Q38Model, Q38ModelOps};
use crate::prompt::ToolParser;
use crate::tokenizer::decoder::ByteDecoder;

use std::sync::mpsc::Sender;

/// Sampling knobs a request may override.
pub struct Req {
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    presence: f32,
    seed: u64,
    thinking: bool,
}

pub enum Event {
    Delta {
        content: String,
        reasoning: String,
    },
    /// Fully parsed `(name, arguments)` pairs decoded out of `<tool_call>`.
    Calls(Vec<(String, String)>),
    Done {
        prompt_tokens: u32,
        out: u32,
        reason: &'static str,
    },
    Fail(String),
}

fn delta(thinking: bool, text: String) -> Event {
    if thinking {
        Event::Delta {
            content: String::new(),
            reasoning: text,
        }
    } else {
        Event::Delta {
            content: text,
            reasoning: String::new(),
        }
    }
}

fn send_text(parser: &mut ToolParser, thinking: bool, text: &str, sink: &Sender<Event>) -> bool {
    let (visible, calls) = parser.feed(text);
    if !visible.is_empty() && sink.send(delta(thinking, visible)).is_err() {
        return false;
    }
    calls.is_empty() || sink.send(Event::Calls(calls)).is_ok()
}

pub fn run(
    model: &mut Q38Model<'_>,
    state: &TokenState,
    decoder: &mut ByteDecoder,
    req: &Req,
    ids: &[u32],
    sink: &Sender<Event>,
) -> Result<(), String> {
    model.reset();
    decoder.reset();
    let mut sampler = Sampler::new(
        req.seed,
        req.temperature,
        req.top_k,
        req.top_p,
        req.presence,
        model.vocab_size() as usize,
    );
    for &id in ids {
        sampler.observe(id);
    }
    let mut thinking = req.thinking;
    let mut out: u32 = 0;
    let mut reason = "length";
    let mut parser = ToolParser::default();

    macro_rules! emit {
        ($tok: expr) => {{
            let tok: u32 = $tok;
            if tok == state.eos {
                reason = "stop";
                false
            } else {
                let text = if Some(tok) == state.think_close {
                    thinking = false; // end thinking
                    String::new()
                } else if state.control.contains(&tok) {
                    String::new()
                } else {
                    decoder.push(tok)
                };
                out += 1;
                if !text.is_empty() && !send_text(&mut parser, thinking, &text, sink) {
                    // client hung up
                    return Ok(());
                }
                true
            }
        }};
    }

    let mut pending = sampler.sample(
        model
            .prefill(ids)
            .map_err(|_| "prefill failed: prompt longer than --context")?,
    );
    let mut accepted = vec![0u32; state.draft as usize + 1];
    loop {
        if !emit!(pending) || out >= req.max_tokens {
            break;
        }
        let room = model.context_length() - model.position();
        if room == 0 {
            break;
        }
        let draft = if req.temperature == 0.0 {
            state.draft.min(req.max_tokens - out).min(room)
        } else {
            0
        };
        if draft == 0 {
            pending = sampler.sample(model.forward_token(pending).map_err(|_| "forward failed")?);
            continue;
        }
        let (_, consumed) = model
            .speculative_greedy(pending, state.eos, draft, &mut accepted[..draft as usize])
            .map_err(|_| "speculative decode failed")?;
        // The last accepted token stays pending; the next round emits it.
        let mut halted = false;
        for &tok in &accepted[..(consumed as usize).saturating_sub(1)] {
            if out >= req.max_tokens || !emit!(tok) {
                halted = true;
                break;
            }
        }
        if halted {
            break;
        }
        pending = accepted[consumed as usize - 1];
    }
    // Flush the byte decoder first, then the tool parser: a call can be split
    // across both buffers, and the parser must see the trailing bytes.
    let tail = decoder.finish();
    if !tail.is_empty() && !send_text(&mut parser, thinking, &tail, sink) {
        return Ok(());
    }
    let (rest, calls) = parser.flush();
    if !rest.is_empty() {
        let _ = sink.send(delta(thinking, rest));
    }
    if !calls.is_empty() {
        let _ = sink.send(Event::Calls(calls));
    }
    let _ = sink.send(Event::Done {
        prompt_tokens: ids.len() as u32,
        out,
        reason,
    });
    Ok(())
}
