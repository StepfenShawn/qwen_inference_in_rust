//! Prompt construction and tool-call parsing.

use serde_json::{Value, json};

/// Reasoning-effort instruction prepended to the system block. These two
/// strings are template text, not invented here.
const EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// The tool preamble is copied verbatim from the checkpoint's own
/// `tokenizer.chat_template`; the model was trained with exactly this text.
const TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOL_RULES: &str = r#"

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// An assistant turn that is about to start reasoning.
const THINKING_OPEN: &str = "<think>\n";
/// An assistant turn that must answer directly: an empty, already-closed block.
const NO_THINKING_OPEN: &str = "<think>\n\n</think>\n\n";

/// How much thinking the system block asks for. `Medium` is what the model does
/// anyway, so it contributes no instruction — matching the template.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Effort {
    Low,
    #[default]
    Medium,
    Xhigh,
}

impl Effort {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "xhigh" => Some(Self::Xhigh),
            _ => None,
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Self::Low => EFFORT_LOW,
            Self::Medium => "",
            Self::Xhigh => EFFORT_XHIGH,
        }
    }
}

/// Renders an OpenAI-style message array exactly as `tokenizer.chat_template`
/// would, and leaves the prompt open at `<|im_start|>assistant\n<think>\n`.
///
/// History that only the server can produce is handled too: `assistant` turns
/// replay their `reasoning_content` and `tool_calls`, and a run of consecutive
/// `tool` results collapses into a single `user` turn of `<tool_response>`
/// blocks — that is what the template does, not a simplification.
pub fn render(messages: &[Value], thinking: bool, effort: Effort, tools: &[Value]) -> String {
    let sys = messages
        .iter()
        .filter(|m| m["role"] == "system")
        .map(|m| text_of(&m["content"]).trim().to_string())
        .collect::<Vec<_>>()
        .join("\n\n");
    let effort = if thinking { effort.instruction() } else { "" };
    let mut out = String::new();
    if !tools.is_empty() {
        // Declare tools first, then the caller's own system text.
        out.push_str("<|im_start|>system\n");
        if !effort.is_empty() {
            out.push_str(effort);
            out.push_str("\n\n");
        }
        out.push_str(TOOLS_HEAD);
        for tool in tools {
            out.push('\n');
            out.push_str(&tool.to_string());
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_RULES);
        if !sys.is_empty() {
            out.push_str("\n\n");
            out.push_str(&sys);
        }
        out.push_str("<|im_end|>\n");
    } else if !effort.is_empty() || !sys.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !effort.is_empty() {
            out.push_str(effort);
            out.push_str(if sys.is_empty() { "" } else { "\n\n" });
        }
        out.push_str(&sys);
        out.push_str("<|im_end|>\n");
    }
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        match message["role"].as_str().unwrap_or("user") {
            "system" => {}
            "tool" => {
                out.push_str("<|im_start|>user");
                while i < messages.len() && messages[i]["role"] == "tool" {
                    out.push_str("\n<tool_response>\n");
                    out.push_str(text_of(&messages[i]["content"]).trim());
                    out.push_str("\n</tool_response>");
                    i += 1;
                }
                out.push_str("<|im_end|>\n");
                continue;
            }
            "assistant" => {
                let content = text_of(&message["content"]).trim().to_string();
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(text_of(&message["reasoning_content"]).trim());
                out.push_str("\n</think>\n\n");
                out.push_str(&content);
                for (n, call) in message["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let function = &call["function"];
                    if n > 0 {
                        out.push('\n');
                    } else if !content.is_empty() {
                        out.push_str("\n\n");
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(function["name"].as_str().unwrap_or(""));
                    out.push_str(">\n");
                    push_args(&mut out, &function["arguments"]);
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            role => {
                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(text_of(&message["content"]).trim());
                out.push_str("<|im_end|>\n");
            }
        }
        i += 1;
    }
    out.push_str("<|im_start|>assistant\n");
    out.push_str(if thinking {
        THINKING_OPEN
    } else {
        NO_THINKING_OPEN
    });
    out
}

/// Single-turn convenience wrapper for the CLI's `--prompt` + `--system`.
pub fn render_simple(user: &str, system: Option<&str>, thinking: bool, effort: Effort) -> String {
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = system.filter(|text| !text.is_empty()) {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": user}));
    render(&messages, thinking, effort, &[])
}

/// OpenAI content is either a string or a list of parts; keep the text parts.
fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect(),
        _ => String::new(),
    }
}

/// OpenAI sends `arguments` as a JSON string; the template wants an object, so
/// both forms end up as `<parameter=K>\nvalue\n</parameter>`.
fn push_args(out: &mut String, arguments: &Value) {
    let parsed = arguments
        .as_str()
        .and_then(|text| serde_json::from_str::<Value>(text).ok());
    let Some(map) = parsed.as_ref().unwrap_or(arguments).as_object() else {
        return;
    };
    for (key, value) in map {
        let text = match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        out.push_str(&format!("<parameter={key}>\n{text}\n</parameter>\n"));
    }
}

/// Streaming splitter: text goes to the caller, `<tool_call>` blocks become
/// `(name, arguments_json)` pairs.
#[derive(Default)]
pub struct ToolParser {
    pub buf: String,
    pub inside: bool,
}

impl ToolParser {
    pub fn feed(&mut self, text: &str) -> (String, Vec<(String, String)>) {
        self.buf.push_str(text);
        let (mut visible, mut calls) = (String::new(), Vec::new());
        loop {
            if self.inside {
                let Some(at) = self.buf.find(CALL_CLOSE) else {
                    break;
                };
                let block = self.buf[..at].to_string();
                self.buf.drain(..at + CALL_CLOSE.len());
                self.inside = false;
                match parse_call(&block) {
                    Some(call) => calls.push(call),
                    // The model can also *quote* the markup in prose; put it back.
                    None => {
                        visible.push_str(CALL_OPEN);
                        visible.push_str(&block);
                        visible.push_str(CALL_CLOSE);
                    }
                }
            } else if let Some(at) = self.buf.find(CALL_OPEN) {
                visible.push_str(&self.buf[..at]);
                self.buf.drain(..at + CALL_OPEN.len());
                self.inside = true;
            } else {
                let cut = self.buf.len() - held_back(&self.buf, CALL_OPEN);
                visible.push_str(&self.buf[..cut]);
                self.buf.drain(..cut);
                break;
            }
        }
        (visible, calls)
    }

    /// End of stream: release what is left, salvaging an unterminated call.
    pub fn flush(&mut self) -> (String, Vec<(String, String)>) {
        let rest = std::mem::take(&mut self.buf);
        if self.inside {
            (String::new(), parse_call(&rest).into_iter().collect())
        } else {
            (rest, Vec::new())
        }
    }
}

/// `<tool_call>\n<function=NAME>\n<parameter=K>\nv\n</parameter>...</function>`
/// becomes `("NAME", {"K": v})`. Values are JSON when they parse as JSON,
/// otherwise they stay strings — `"3"` becomes a number, `"/tmp"` stays text.
fn parse_call(block: &str) -> Option<(String, String)> {
    let start = block.find("<function=")? + "<function=".len();
    let end = block[start..].find('>')? + start;
    let name = block[start..end].trim().to_string();
    let mut args = serde_json::Map::new();
    let mut rest = &block[end..];
    while let Some(at) = rest.find("<parameter=") {
        let key_start = at + "<parameter=".len();
        let Some(gt) = rest[key_start..].find('>') else {
            break;
        };
        let key = rest[key_start..key_start + gt].trim().to_string();
        let value_start = key_start + gt + 1;
        let Some(close) = rest[value_start..].find("</parameter>") else {
            break;
        };
        let raw = rest[value_start..value_start + close].trim();
        args.insert(
            key,
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())),
        );
        rest = &rest[value_start + close + "</parameter>".len()..];
    }
    Some((name, Value::Object(args).to_string()))
}

/// Longest suffix of `text` that is a prefix of `marker`. A tokenizer can cut
/// `<tool_cal| l>` anywhere, so this is how much text must be held back.
fn held_back(text: &str, marker: &str) -> usize {
    let max = (marker.len() - 1).min(text.len());
    (1..=max)
        .rev()
        .find(|&n| text.ends_with(&marker[..n]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        "I will read it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a.rs\n</parameter>\n<parameter=lines>\n3\n</parameter>\n</function>\n</tool_call>"
    }

    #[test]
    fn prompt_follows_the_checkpoint_template() {
        let tools = [
            json!({"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}),
        ];
        let messages = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "read a.rs"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}}]},
            {"role": "tool", "content": "fn main() {}"},
            {"role": "tool", "content": "second"}
        ]);
        let prompt = render(messages.as_array().unwrap(), true, Effort::Xhigh, &tools);
        assert!(prompt.contains("<tools>\n{\"function\":"), "{prompt}");
        assert!(prompt.contains("<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call><|im_end|>"), "{prompt}");
        assert!(prompt.contains("<|im_start|>user\n<tool_response>\nfn main() {}\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>"), "{prompt}");
        assert!(prompt.contains("\n\nbe brief<|im_end|>"), "{prompt}");
        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "{prompt}"
        );
    }

    /// The CLI path must stay byte-identical to the old hand-rolled renderer.
    #[test]
    fn simple_prompt_matches_the_cli_history() {
        let expected = format!(
            "<|im_start|>system\n{EFFORT_XHIGH}\n\nbe terse<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            render_simple("hi", Some("be terse"), true, Effort::Xhigh),
            expected
        );

        // --no-thinking: no system block, and the reasoning block is pre-closed.
        assert_eq!(
            render_simple("hi", None, false, Effort::Xhigh),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        // medium/low: only `low` adds an instruction.
        assert_eq!(
            render_simple("hi", Some("s"), true, Effort::Medium),
            "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert!(render_simple("hi", None, true, Effort::Low).contains(EFFORT_LOW));
    }

    #[test]
    fn parses_name_and_arguments() {
        let (name, args) = parse_call(sample().split("<tool_call>").nth(1).unwrap()).unwrap();
        assert_eq!(name, "read_file");
        assert_eq!(
            serde_json::from_str::<Value>(&args).unwrap(),
            json!({"path": "src/a.rs", "lines": 3})
        );
    }

    #[test]
    fn markers_never_leak_into_content() {
        // One character at a time is the worst case for a tokenizer split.
        let mut parser = ToolParser::default();
        let (mut visible, mut calls) = (String::new(), Vec::new());
        for ch in sample().chars() {
            let (text, found) = parser.feed(&ch.to_string());
            visible.push_str(&text);
            calls.extend(found);
        }
        let (tail, found) = parser.flush();
        visible.push_str(&tail);
        calls.extend(found);
        assert_eq!(visible, "I will read it.\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "read_file");
    }

    #[test]
    fn partial_marker_is_held_back() {
        let mut parser = ToolParser::default();
        assert_eq!(parser.feed("answer <tool_c").0, "answer ");
        assert!(parser.feed("all>").0.is_empty());
        assert!(parser.inside);
    }

    #[test]
    fn quoted_markup_stays_in_the_text() {
        let mut parser = ToolParser::default();
        let (visible, calls) = parser.feed("a <tool_call>not a call</tool_call> b");
        assert_eq!(visible, "a <tool_call>not a call</tool_call> b");
        assert!(calls.is_empty());
    }

    #[test]
    fn flush_salvages_an_unterminated_call() {
        let mut parser = ToolParser::default();
        assert_eq!(
            parser
                .feed("ok<tool_call>\n<function=ls>\n<parameter=dir>\n/tmp\n</parameter>")
                .0,
            "ok"
        );
        let (rest, calls) = parser.flush();
        assert!(rest.is_empty());
        assert_eq!(calls[0].0, "ls");
        assert_eq!(calls[0].1, "{\"dir\":\"/tmp\"}");
    }
}
