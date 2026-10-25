//! `qwen_inference_server` — an OpenAI-compatible HTTP front end for the Qwen3.8
//! 27B engine. No async runtime, no web framework: one blocking `TcpListener`, a
//! thread per connection, and the model driven from a single thread so the checkpoint
//! is never shared.

use qwen38_27b_in_rust::gguf::Gguf;
use qwen38_27b_in_rust::kernel::Q38Iq1sRepack;
use qwen38_27b_in_rust::model::sampler::Sampler;
use qwen38_27b_in_rust::model::{Q38Model, Q38ModelOps};
use qwen38_27b_in_rust::prompt::{Effort, ToolParser, render};
use qwen38_27b_in_rust::tokenizer::{self, ByteDecoder};

use serde_json::{Value, json};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Sender, channel};
use std::time::{SystemTime, UNIX_EPOCH};

const USAGE: &str = "\
usage: qwen_inference_server --model MODEL.gguf [--host 127.0.0.1] [--port 8080]
  --context N     context capacity (default: 4096)
  --threads N     worker threads for the kernels
  --mtp-draft N   greedy speculative decoding, temperature 0 only (0 = off)
  routes: POST /v1/chat/completions, GET /v1/models, GET /health";

/// Model-independent state, built once and read by every request.
struct Engine {
    eos: u32,
    think_close: Option<u32>,
    control: HashSet<u32>,
    draft: u32,
}

struct Job {
    prompt: String,
    req: Req,
    sink: Sender<Event>,
}

/// Sampling knobs a request may override.
struct Req {
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    presence: f32,
    seed: u64,
    thinking: bool,
}

enum Event {
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

macro_rules! reply {
    ($w:expr, $status:expr, $body:expr) => {{
        let body: String = $body.to_string();
        write!(
            $w,
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
            Access-Control-Allow-Origin: *\r\n\r\n{}",
            $status,
            body.len(),
            body
        )
        .and_then(|_| $w.flush())
        .is_ok()
    }};
}

macro_rules! sse {
    ($w:expr, $value:expr) => {
        write!($w, "data: {}\n\n", $value)
            .and_then(|_| $w.flush())
            .is_ok()
    };
}

#[inline]
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
}

fn push_text(parser: &mut ToolParser, thinking: bool, text: &str, sink: &Sender<Event>) -> bool {
    let (visible, calls) = parser.feed(text);
    if !visible.is_empty() && sink.send(delta(thinking, visible)).is_err() {
        return false;
    }
    calls.is_empty() || sink.send(Event::Calls(calls)).is_ok()
}

/// One generation, on the model thread. `Ok(())` also covers "client hung up",
/// which is signalled by a failed send on the sink.
fn run(
    model: &mut Q38Model<'_>,
    engine: &Engine,
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
    // Before the first `</think>` the text is reasoning, afterwards it is answer.
    let (mut thinking, mut out, mut reason) = (req.thinking, 0u32, "length");
    let mut parser = ToolParser::default();
    macro_rules! emit {
        ($tok:expr) => {{
            let tok: u32 = $tok;
            if tok == engine.eos {
                reason = "stop";
                false
            } else {
                let text = if Some(tok) == engine.think_close {
                    thinking = false;
                    String::new()
                } else if engine.control.contains(&tok) {
                    String::new()
                } else {
                    decoder.push(tok)
                };
                out += 1;
                if !text.is_empty() && !push_text(&mut parser, thinking, &text, sink) {
                    return Ok(()); // client hung up: stop burning cycles
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
    let mut accepted = vec![0u32; engine.draft as usize + 1];
    loop {
        if !emit!(pending) || out >= req.max_tokens {
            break;
        }
        let room = model.context_length() - model.position();
        if room == 0 {
            break;
        }
        // `speculative_greedy` is argmax only, so MTP is limited to temperature 0.
        let draft = if req.temperature == 0.0 {
            engine.draft.min(req.max_tokens - out).min(room)
        } else {
            0
        };
        if draft == 0 {
            pending = sampler.sample(model.forward_token(pending).map_err(|_| "forward failed")?);
            continue;
        }
        let (_, consumed) = model
            .speculative_greedy(pending, engine.eos, draft, &mut accepted[..draft as usize])
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
    if !tail.is_empty() && !push_text(&mut parser, thinking, &tail, sink) {
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

/// Reads one HTTP/1.1 request: `(method, path, body)`.
fn read_http(rd: &mut BufReader<TcpStream>) -> Option<(String, String, Vec<u8>)> {
    let (mut head, mut len) = (String::new(), 0usize);
    loop {
        let mut line = String::new();
        if rd.read_line(&mut line).ok()? == 0 {
            return None;
        }
        if line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            len = value.trim().parse().ok()?;
        }
        head.push_str(&line);
    }
    let mut body = vec![0u8; len];
    rd.read_exact(&mut body).ok()?;
    let mut parts = head.lines().next()?.split_whitespace();
    Some((parts.next()?.to_string(), parts.next()?.to_string(), body))
}

fn serve(rd: &mut BufReader<TcpStream>, jobs: &Sender<Job>, name: &str) {
    while let Some((method, path, body)) = read_http(rd) {
        let route = (method.as_str(), path.split('?').next().unwrap_or("/"));
        let ok = match route {
            ("GET", "/health") => reply!(
                rd.get_mut(),
                "200 OK",
                json!({"status": "ok", "model": name})
            ),
            ("GET", "/v1/models") => reply!(
                rd.get_mut(),
                "200 OK",
                json!({"object": "list", "data": [{"id": name, "object": "model"}]})
            ),
            ("POST", "/v1/chat/completions") => chat(rd.get_mut(), jobs, name, &body),
            _ => reply!(
                rd.get_mut(),
                "404 Not Found",
                json!({"error": {"message": format!("no route for {method} {path}")}})
            ),
        };
        if !ok {
            return;
        }
    }
}

/// Handles one request; `false` means "close this connection".
fn chat(w: &mut TcpStream, jobs: &Sender<Job>, name: &str, body: &[u8]) -> bool {
    let Ok(parsed) = serde_json::from_slice::<Value>(body) else {
        return reply!(
            w,
            "400 Bad Request",
            json!({"error": {"message": "body is not JSON"}})
        );
    };
    let Some(messages) = parsed["messages"].as_array() else {
        return reply!(
            w,
            "400 Bad Request",
            json!({"error": {"message": "messages[] is required"}})
        );
    };
    let (id, created, stream) = (
        format!("chatcmpl-{:x}", now()),
        now() / 1_000_000_000,
        parsed["stream"].as_bool().unwrap_or(false),
    );
    let thinking = parsed["thinking"].as_bool().unwrap_or(true);
    // `tool_choice: "none"` keeps the declarations out of the prompt entirely.
    let tools: Vec<Value> = match parsed["tools"].as_array() {
        Some(list) if parsed["tool_choice"] != "none" => list.clone(),
        _ => Vec::new(),
    };
    let prompt = render(messages, thinking, Effort::Xhigh, &tools);
    println!("{}", prompt);
    let req = Req {
        max_tokens: parsed["max_tokens"].as_u64().unwrap_or(1024) as u32,
        temperature: parsed["temperature"].as_f64().unwrap_or(1.0) as f32,
        top_k: parsed["top_k"].as_u64().unwrap_or(20) as u32,
        top_p: parsed["top_p"].as_f64().unwrap_or(0.95) as f32,
        presence: parsed["presence_penalty"].as_f64().unwrap_or(0.0) as f32,
        seed: parsed["seed"].as_u64().unwrap_or_else(now),
        thinking,
    };
    let (tx, rx) = channel();
    if jobs
        .send(Job {
            prompt,
            req,
            sink: tx,
        })
        .is_err()
    {
        return false;
    }
    if stream && write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").and_then(|_| w.flush()).is_err() {
        return false;
    }
    let chunk = |delta: Value, finish: Value| {
        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let (mut text, mut reasoning) = (String::new(), String::new());
    let mut calls: Vec<Value> = Vec::new();
    let (mut prompt_tokens, mut out, mut reason) = (0u32, 0u32, "stop");
    for event in rx {
        match event {
            Event::Delta {
                content,
                reasoning: think,
            } if stream => {
                let delta = if think.is_empty() {
                    json!({"content": content})
                } else {
                    json!({"reasoning_content": think})
                };
                if !sse!(w, chunk(delta, Value::Null)) {
                    return false;
                }
            }
            Event::Delta {
                content,
                reasoning: think,
            } => {
                text.push_str(&content);
                reasoning.push_str(&think);
            }
            Event::Calls(list) => {
                for (name, arguments) in list {
                    let call = json!({"id": format!("call_{:x}", now() + calls.len() as u64),
                        "type": "function", "function": {"name": name, "arguments": arguments}});
                    if stream {
                        let mut delta = call.clone();
                        delta["index"] = json!(calls.len());
                        if !sse!(w, chunk(json!({"tool_calls": [delta]}), Value::Null)) {
                            return false;
                        }
                    }
                    calls.push(call);
                }
            }
            Event::Done {
                prompt_tokens: p,
                out: o,
                reason: r,
            } => (prompt_tokens, out, reason) = (p, o, r),
            Event::Fail(message) => {
                if stream {
                    let _ = sse!(w, json!({"error": {"message": message}}));
                } else {
                    let _ = reply!(
                        w,
                        "500 Internal Server Error",
                        json!({"error": {"message": message}})
                    );
                }
                return false;
            }
        }
    }
    if !calls.is_empty() {
        reason = "tool_calls";
    }
    if stream {
        let _ = sse!(w, chunk(json!({}), json!(reason)));
        let _ = write!(w, "data: [DONE]\n\n");
        let _ = w.flush();
        return false;
    }
    let mut message = json!({"role": "assistant", "content": text, "reasoning_content": reasoning});
    if !calls.is_empty() {
        if text.is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = json!(calls);
    }
    reply!(
        w,
        "200 OK",
        json!({"id": id, "object": "chat.completion", "created": created, "model": name,
        "choices": [{"index": 0, "finish_reason": reason, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": out, "total_tokens": prompt_tokens + out}})
    )
}

fn parse_args() -> Result<(String, String, u16, u32, Option<usize>, u32), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut host, mut port) = (None::<String>, "127.0.0.1".to_string(), 8080u16);
    let (mut context, mut threads, mut draft) = (4096u32, None::<usize>, 0u32);
    let mut i = 0;
    while i < argv.len() {
        let value = || {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--model" => model = Some(value()?),
            "--host" => host = value()?,
            "--port" => port = value()?.parse().map_err(|_| "invalid --port")?,
            "--context" => context = value()?.parse().map_err(|_| "invalid --context")?,
            "--threads" => threads = Some(value()?.parse().map_err(|_| "invalid --threads")?),
            "--mtp-draft" => draft = value()?.parse().map_err(|_| "invalid --mtp-draft")?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 2;
    }
    Ok((
        model.ok_or("--model is required")?,
        host,
        port,
        context,
        threads,
        draft,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (path, host, port, context, threads, draft) = parse_args()?;
    if let Some(count) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .build_global()?;
    }
    let mut gguf = Gguf::open(&path)?;
    if let Err(error) = gguf.prepare_iq1_s_repacks() {
        eprintln!("server: IQ1 runtime repack unavailable; using packed weights ({error})");
    }
    let (tok, mut decoder) = tokenizer::build_tokenizer_and_decoder_from_gguf(&gguf)?;
    let engine = Engine {
        eos: gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(0),
        think_close: tok.token_to_id("</think>"),
        control: tokenizer::get_all_control_token_ids(&gguf)
            .into_iter()
            .collect(),
        draft,
    };
    let mut model = Q38Model::open_gguf(&gguf, context).map_err(|_| "unable to open model")?;
    if draft > 0 {
        model
            .enable_mtp()
            .map_err(|_| "MTP head unavailable: checkpoint must carry blk.64.nextn.*")?;
    }
    let name = std::path::Path::new(&path)
        .file_name()
        .map(|n| n.to_string_lossy().trim_end_matches(".gguf").to_string())
        .unwrap_or_else(|| "local".into());

    let listener = TcpListener::bind((host.as_str(), port))?;
    println!("server: http://{host}:{port} model={name} context={context} draft={draft}");
    let (job_tx, job_rx) = channel::<Job>();
    let (accept_tx, accept_name) = (job_tx.clone(), name.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (jobs, model_name) = (accept_tx.clone(), accept_name.clone());
            std::thread::spawn(move || {
                let _ = stream.set_nodelay(true);
                serve(&mut BufReader::new(stream), &jobs, &model_name);
            });
        }
    });

    // The model thread: requests are decoded one at a time, which is also the
    // bandwidth-optimal schedule for a single checkpoint.
    for job in job_rx {
        let ids = match tok.encode(job.prompt.as_str(), false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(error) => {
                let _ = job.sink.send(Event::Fail(format!("tokenizer: {error}")));
                continue;
            }
        };
        println!("Input toks: {}", ids.len());
        if let Err(message) = run(&mut model, &engine, &mut decoder, &job.req, &ids, &job.sink) {
            let _ = job.sink.send(Event::Fail(message));
        }
    }
    Ok(())
}
