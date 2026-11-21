use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Sender, channel};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::prompt::{self, Effort};
use crate::server::{Event, Req};

pub struct Job {
    pub prompt: String,
    pub req: Req,
    pub sink: Sender<Event>,
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

/// Reads one HTTP/1.1 request: `(method, path, body)`.
fn read_http(rd: &mut BufReader<TcpStream>) -> Option<(String, String, Vec<u8>)> {
    let mut head = String::new();
    let mut len = 0;
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
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    Some((method, path, body))
}

pub fn serve(rd: &mut BufReader<TcpStream>, jobs: &Sender<Job>, name: &str) {
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
    let id = format!("chatcmpl-{:x}", now());
    let created = now() / 1_000_000_000;
    let stream = parsed["stream"].as_bool().unwrap_or(false);
    let thinking = parsed["thinking"].as_bool().unwrap_or(true);
    // `tool_choice: "none"` keeps the declarations out of the prompt entirely.
    let tools: Vec<Value> = match parsed["tools"].as_array() {
        Some(list) if parsed["tool_choice"] != "none" => list.clone(),
        _ => Vec::new(),
    };
    let prompt = prompt::render(messages, thinking, Effort::Xhigh, &tools);

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
    if stream {
        if write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")
        .and_then(|_| w.flush()).is_err() {
            return false;
        }
    }
    let chunk = |delta: Value, finish: Value| {
        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut prompt_tokens: u32 = 0;
    let mut out: u32 = 0;
    let mut reason = "stop";

    for event in rx {
        match event {
            Event::Delta {
                content,
                reasoning: think,
            } if stream => {
                let delta = if think.is_empty() {
                    // no-thinking
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
