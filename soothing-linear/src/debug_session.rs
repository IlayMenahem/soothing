//! Session-scoped NDJSON debug logging (Cursor debug mode). Not part of the public API.

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const DEBUG_LOG_PATH: &str =
    "/Users/ilaymenahem/Documents/university.nosync/Reasearch/soothing/.cursor/debug-c27c26.log";
const SESSION_ID: &str = "c27c26";

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Append one NDJSON line for the active debug session.
pub(crate) fn log_ndjson(
    location: &str,
    message: &str,
    hypothesis_id: &str,
    data_fields: &[(&str, &str)],
) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut data = String::from('{');
    for (i, (k, v)) in data_fields.iter().enumerate() {
        if i > 0 {
            data.push(',');
        }
        data.push_str(&format!("{}:{}", json_string(k), json_string(v)));
    }
    data.push('}');
    let line = format!(
        "{{\"sessionId\":{},\"timestamp\":{},\"location\":{},\"message\":{},\"hypothesisId\":{},\"data\":{}}}\n",
        json_string(SESSION_ID),
        ts,
        json_string(location),
        json_string(message),
        json_string(hypothesis_id),
        data
    );
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(DEBUG_LOG_PATH)
    {
        let _ = f.write_all(line.as_bytes());
    }
}
