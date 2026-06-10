//! Text-format BlockPar parser — ports `CBlockPar::LoadFromText` and its
//! `BPCompiler` (MatrixLib/Base/src/CBlockPar.cpp:878-1120).
//!
//! Grammar (distilled from `BPCompiler::Run`):
//!
//! ```text
//! // line-comment (stripped before parsing)
//! Name = Value                  // param
//! Value                         // anonymous param (empty name)
//! Name {                        // block open; body on next line(s)
//!   ...
//! }
//! Name { Value }                // inline block (opens and closes on one line)
//! Name -sort { ... }            // block options: -sort / +sort toggle
//! Name = path/to/file.txt { }   // =<file> hint (stored, not auto-resolved
//!                               //   here — consumers open the file if needed)
//! ```
//!
//! Line terminators are `\n`, `\r`, or `\r\n`. Only `//` EOL comments are
//! recognized; `/* */` is not supported by the original grammar.
//!
//! The original supports both UTF-16LE (0xFEFF BOM) and ANSI source text
//! (`CBlockPar.cpp:1128-1155`). `decode_source` handles both.
//!
//! The parser does not open `=<file>` hints itself: the original calls
//! `LoadFromTextFile` against the game's `CFile`, which is an asset loader
//! we don't have here. Instead, the filename is parsed and kept on the
//! block. Callers that want C++-equivalent eager inclusion pass a loader
//! callback to `BlockPar::resolve_includes` (or the `parse_bytes_with_loader`
//! convenience wrapper), which walks the tree and fills each `=<file>`
//! block in place.

use std::collections::HashMap;

/// One entry inside a block, in source order. Matches `CBlockParUnit`
/// (CBlockPar.hpp).
#[derive(Debug, Clone)]
pub enum Entry {
    /// `m_Type == 1` in the C++. Stored with its source-order name; an empty
    /// name corresponds to an anonymous (non-`Name = ...`) value line.
    Par { name: String, value: String },
    /// `m_Type == 2` in the C++.
    Block {
        name: String,
        block: BlockPar,
        /// `=<file>` hint when the block header carried one (see
        /// `BPCompiler::WordFindOption`, CBlockPar.cpp:1029-1040).
        from_file: Option<String>,
    },
    /// `m_Type == 0` — a standalone comment line. Rarely needed by
    /// consumers but preserved for round-trip fidelity.
    Comment(String),
}

/// Ports `CBlockPar`. Entries are kept in source order; `name_index` gives
/// the `ArrayFind` fast path used by the original's sorted variant.
#[derive(Debug, Clone, Default)]
pub struct BlockPar {
    entries: Vec<Entry>,
    /// Maps lowercased name → every entry index with that name, in source
    /// order. Case-insensitive to mirror the CWStr comparison used by
    /// `ParGetNE_` (CBlockPar.cpp:412-432), which treats `Equal` as an ASCII
    /// comparison in practice for our key space.
    name_index: HashMap<String, Vec<usize>>,
    /// Tracks the `+sort` / `-sort` switch parsed from the block header. The
    /// Rust runtime doesn't reorder entries based on this flag — consumers
    /// iterate in source order — but we preserve it so a future sorted view
    /// can use it.
    sort: bool,
}

impl BlockPar {
    /// Parse a raw source buffer. Auto-detects UTF-16LE (0xFEFF BOM) vs
    /// ANSI/UTF-8. Always succeeds: invalid bytes degrade to replacement
    /// characters, matching how the original drops malformed wide chars.
    pub fn parse_bytes(bytes: &[u8]) -> Self {
        Self::parse(&decode_source(bytes))
    }

    /// Parse a pre-decoded string.
    pub fn parse(text: &str) -> Self {
        // The original treats the root block as sorted by default
        // (`CBlockPar` ctor sets `m_Sort = true` — see CBlockPar.hpp).
        let mut root = BlockPar {
            sort: true,
            ..BlockPar::default()
        };
        let chars: Vec<char> = text.chars().collect();
        let mut cursor = 0;
        parse_block_body(&chars, &mut cursor, &mut root, /*nested=*/ false);
        root
    }

    /// Parse and eagerly resolve every `=<file>` include. Equivalent to
    /// `parse_bytes` followed by `resolve_includes` — ports the combined
    /// effect of `LoadFromText` + `LoadFromTextFile` (CBlockPar.cpp:1072-1073).
    pub fn parse_bytes_with_loader<F>(bytes: &[u8], mut load: F) -> Self
    where
        F: FnMut(&str) -> Option<Vec<u8>>,
    {
        let mut bp = Self::parse_bytes(bytes);
        bp.resolve_includes(&mut load);
        bp
    }

    /// Walk the tree and, for every block carrying a `from_file` hint, call
    /// `load(path)` to fetch the referenced file, parse it, and splice its
    /// entries into the block. Ports the eager inclusion in
    /// CBlockPar.cpp:1072 (`m_Block->LoadFromTextFile(...)`). The
    /// `from_file` hint is kept on the block so save/round-trip still sees
    /// it. Loaded blocks are themselves resolved so nested `=<file>`
    /// references work recursively, matching the original's recursive
    /// `LoadFromText`.
    pub fn resolve_includes<F>(&mut self, load: &mut F)
    where
        F: FnMut(&str) -> Option<Vec<u8>>,
    {
        for entry in &mut self.entries {
            if let Entry::Block {
                block, from_file, ..
            } = entry
            {
                if let Some(path) = from_file.as_ref() {
                    if let Some(bytes) = load(path) {
                        let loaded = BlockPar::parse_bytes(&bytes);
                        block.entries = loaded.entries;
                        block.name_index = loaded.name_index;
                        block.sort = loaded.sort;
                    }
                }
                block.resolve_includes(load);
            }
        }
    }

    /// Number of entries (all kinds). Mirrors `CBlockPar::AllCount`.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Ports `ParGetNE_(name, 0)` (CBlockPar.cpp:412-432): returns the first
    /// param with this name, or `None` if absent. Name lookup is case
    /// insensitive; case-sensitive lookups should use `entries()` directly.
    pub fn par_get_ne(&self, name: &str) -> Option<&str> {
        let key = name.to_ascii_lowercase();
        // The original scans all same-named entries and returns the first
        // *unit* (m_Type==1), skipping blocks that share the name.
        self.name_index
            .get(&key)?
            .iter()
            .find_map(|&idx| match &self.entries[idx] {
                Entry::Par { value, .. } => Some(value.as_str()),
                _ => None,
            })
    }

    /// Ports `BlockGet(name)` — returns the first child block by name.
    pub fn block_get(&self, name: &str) -> Option<&BlockPar> {
        let key = name.to_ascii_lowercase();
        let indices = self.name_index.get(&key)?;
        for &i in indices {
            if let Entry::Block { block, .. } = &self.entries[i] {
                return Some(block);
            }
        }
        None
    }

    fn push_entry(&mut self, entry: Entry) {
        let idx = self.entries.len();
        let key = match &entry {
            Entry::Par { name, .. } | Entry::Block { name, .. } => Some(name.to_ascii_lowercase()),
            Entry::Comment(_) => None,
        };
        self.entries.push(entry);
        if let Some(k) = key {
            self.name_index.entry(k).or_default().push(idx);
        }
    }
}

/// Decode a BlockPar source buffer. Detects the UTF-16LE BOM (0xFEFF)
/// exactly as `CBlockPar::LoadFromTextFile` does (CBlockPar.cpp:1128-1155):
/// the first two bytes read as a little-endian u16 compare against 0xFEFF.
/// Anything else is treated as 8-bit source; we use lossy UTF-8 decoding so
/// the ASCII subset used by shipped files (keys like `AlphaTest`, numeric
/// values) round-trips cleanly.
fn decode_source(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        let body = &bytes[2..];
        let mut out = String::with_capacity(body.len() / 2);
        let mut iter = body.chunks_exact(2);
        let mut units: Vec<u16> = Vec::with_capacity(body.len() / 2);
        for c in iter.by_ref() {
            units.push(u16::from_le_bytes([c[0], c[1]]));
        }
        // Decode surrogate-aware; invalid units become U+FFFD to match the
        // lenient behavior of the original when faced with damaged wide text.
        for ch in char::decode_utf16(units.iter().copied()) {
            out.push(ch.unwrap_or(char::REPLACEMENT_CHARACTER));
        }
        out
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

fn is_space(ch: char) -> bool {
    ch == ' ' || ch == '\t'
}

fn is_line_end(ch: char) -> bool {
    ch == '\n' || ch == '\r'
}

/// Skip spaces/tabs on the current line only.
fn skip_inline_space(chars: &[char], cursor: &mut usize) {
    while *cursor < chars.len() && is_space(chars[*cursor]) {
        *cursor += 1;
    }
}

/// Find the `//` comment start within `[cursor, line_end)` and return the
/// effective content end (position just past the last non-space char before
/// the `//`, or `line_end` if no comment). Ports `FindCom`
/// (CBlockPar.cpp:928-937).
fn trim_trailing_spaces(chars: &[char], start: usize, mut end: usize) -> usize {
    while end > start && is_space(chars[end - 1]) {
        end -= 1;
    }
    end
}

fn find_comment(chars: &[char], line_start: usize, line_end: usize) -> (usize, usize) {
    // Returns (comment_start, content_end). comment_start == line_end when
    // there's no //.
    let mut i = line_start;
    while i + 1 < line_end {
        if chars[i] == '/' && chars[i + 1] == '/' {
            let content_end = trim_trailing_spaces(chars, line_start, i);
            return (i, content_end);
        }
        i += 1;
    }
    (line_end, line_end)
}

fn advance_to_line_end(chars: &[char], cursor: usize) -> usize {
    let mut i = cursor;
    while i < chars.len() && !is_line_end(chars[i]) {
        i += 1;
    }
    i
}

/// Skip `\r`, `\n`, or `\r\n` once. Returns the new cursor, or None when
/// at EOF.
fn skip_line_terminator(chars: &[char], cursor: &mut usize) -> bool {
    if *cursor >= chars.len() {
        return false;
    }
    if chars[*cursor] == '\r' {
        *cursor += 1;
        if *cursor < chars.len() && chars[*cursor] == '\n' {
            *cursor += 1;
        }
        true
    } else if chars[*cursor] == '\n' {
        *cursor += 1;
        true
    } else {
        false
    }
}

/// Parse the body of a block. `nested` means we're inside a `{}` and should
/// stop at the matching `}`. The root parse call passes `nested=false`, so
/// a stray `}` at the top level is reported as a comment line rather than
/// silently consumed.
fn parse_block_body(chars: &[char], cursor: &mut usize, block: &mut BlockPar, nested: bool) {
    loop {
        // Start-of-line: skip leading spaces on this line so we can check
        // for `}` at the indentation column.
        skip_inline_space(chars, cursor);
        if *cursor >= chars.len() {
            return;
        }

        // End of the enclosing block.
        if nested && chars[*cursor] == '}' {
            *cursor += 1;
            return;
        }

        // Empty / comment-only line: skip.
        if is_line_end(chars[*cursor]) {
            skip_line_terminator(chars, cursor);
            continue;
        }

        let line_start = *cursor;
        let raw_line_end = advance_to_line_end(chars, line_start);
        let (_comment_start, content_end) = find_comment(chars, line_start, raw_line_end);

        if line_start >= content_end {
            // Pure-comment line.
            let comment_text: String = chars[line_start..raw_line_end].iter().collect();
            block.push_entry(Entry::Comment(comment_text));
            *cursor = raw_line_end;
            skip_line_terminator(chars, cursor);
            continue;
        }

        // Mirror BPCompiler::Run's dispatch order (CBlockPar.cpp:1012-1098):
        // `FindBlock` runs first, so any line with `{` is a block
        // declaration — an `=` before the brace is parsed as a block-header
        // option (e.g. `Name =path/to/file.txt { }`), not a param.
        let brace_pos = find_char(chars, line_start, content_end, '{');
        if let Some(bp) = brace_pos {
            parse_block_declaration(chars, cursor, block, line_start, content_end, bp);
            continue;
        }

        let equals_pos = find_char(chars, line_start, content_end, '=');
        if let Some(eq) = equals_pos {
            parse_named_param(chars, block, line_start, content_end, eq);
        } else {
            // Anonymous value line — whole line stored as value with an
            // empty name (matches `else` branch at CBlockPar.cpp:1088-1098).
            let value: String = chars[line_start..content_end].iter().collect();
            block.push_entry(Entry::Par {
                name: String::new(),
                value: value.trim().to_string(),
            });
        }
        *cursor = raw_line_end;
        skip_line_terminator(chars, cursor);
    }
}

fn find_char(chars: &[char], start: usize, end: usize, target: char) -> Option<usize> {
    (start..end).find(|&i| chars[i] == target)
}

fn parse_named_param(
    chars: &[char],
    block: &mut BlockPar,
    line_start: usize,
    content_end: usize,
    eq: usize,
) {
    // Name = chars in [line_start, eq), right-trimmed.
    let name_end = trim_trailing_spaces(chars, line_start, eq);
    let name: String = chars[line_start..name_end].iter().collect();
    // Value = chars in (eq, content_end], both-trimmed.
    let val_start = eq + 1;
    let value_raw: String = chars[val_start..content_end].iter().collect();
    block.push_entry(Entry::Par {
        name: name.trim().to_string(),
        value: value_raw.trim().to_string(),
    });
}

fn parse_block_declaration(
    chars: &[char],
    cursor: &mut usize,
    parent: &mut BlockPar,
    line_start: usize,
    content_end: usize,
    brace_pos: usize,
) {
    // Header runs from line_start up to brace_pos. Options (e.g. `-sort`,
    // `+sort`, `=<file>`) are parsed after the block name.
    let header_end = brace_pos;

    // Name: first whitespace-separated token that isn't `-`/`=` (see
    // `WordFindBlockName`, CBlockPar.cpp:962-971).
    let mut i = line_start;
    while i < header_end && is_space(chars[i]) {
        i += 1;
    }
    let mut name = String::new();
    if i < header_end && chars[i] != '-' && chars[i] != '=' {
        let name_start = i;
        while i < header_end && !is_space(chars[i]) && chars[i] != '=' {
            i += 1;
        }
        name = chars[name_start..i].iter().collect();
    }

    // Remaining options.
    let mut sort = parent.sort;
    let mut from_file: Option<String> = None;
    while i < header_end {
        while i < header_end && is_space(chars[i]) {
            i += 1;
        }
        if i >= header_end {
            break;
        }
        let tok_start = i;
        while i < header_end && !is_space(chars[i]) {
            i += 1;
        }
        let tok: String = chars[tok_start..i].iter().collect();
        if tok.eq_ignore_ascii_case("-sort") {
            sort = false;
        } else if tok.eq_ignore_ascii_case("+sort") {
            sort = true;
        } else if let Some(path) = tok.strip_prefix('=') {
            from_file = Some(path.to_string());
        }
        // Unknown options mirror the original's silent ignore path.
    }

    // Create the inner block, then parse its body.
    let mut inner = BlockPar {
        entries: Vec::new(),
        name_index: HashMap::new(),
        sort,
    };

    // Check for the single-line `{ ... }` form. The opening `{` is at
    // `brace_pos`. Look for a matching `}` on the same line inside
    // `(brace_pos, content_end)`.
    let inline_close = find_char(chars, brace_pos + 1, content_end, '}');

    match inline_close {
        Some(close) => {
            // Parse the inline body from after `{` to `}` as if it were its
            // own line. This mirrors the inline-`{...}` behavior where the
            // original calls `IsEmptyBlock` and then continues the same
            // line — we just recursively dispatch.
            let mut inner_cursor = brace_pos + 1;
            parse_block_body_inline_range(chars, &mut inner_cursor, &mut inner, close);
            *cursor = close + 1;
            // Advance to end-of-line so we consume trailing comments and the
            // newline.
            *cursor = advance_to_line_end(chars, *cursor);
            skip_line_terminator(chars, cursor);
        }
        None => {
            // Multi-line block: body starts on the line after `{`.
            *cursor = brace_pos + 1;
            // Skip rest of this line (may be comment or nothing).
            *cursor = advance_to_line_end(chars, *cursor);
            skip_line_terminator(chars, cursor);
            parse_block_body(chars, cursor, &mut inner, /*nested=*/ true);
            // After `}` was consumed by the recursive call, advance past any
            // trailing content on the closing-brace line.
            *cursor = advance_to_line_end(chars, *cursor);
            skip_line_terminator(chars, cursor);
        }
    }

    parent.push_entry(Entry::Block {
        name: name.trim().to_string(),
        block: inner,
        from_file,
    });
}

/// Parse block content that lives on a single line between `{` and `}`.
/// This is narrower than `parse_block_body` because it doesn't need to hunt
/// for newlines or the closing brace — the caller already knows where the
/// body ends.
fn parse_block_body_inline_range(
    chars: &[char],
    cursor: &mut usize,
    block: &mut BlockPar,
    body_end: usize,
) {
    // Trim leading whitespace.
    while *cursor < body_end && is_space(chars[*cursor]) {
        *cursor += 1;
    }
    if *cursor >= body_end {
        return;
    }

    // Within a single-line body we allow a param, anonymous value, or a
    // nested inline block. Blocks with multi-line bodies are disallowed
    // here because there's no newline to scan for — that matches the
    // original: `IsEmptyBlock` is the only inline-brace form, anything else
    // would already have started a multi-line body.
    // Same dispatch ordering as `parse_block_body`: a `{` takes priority
    // over `=` so nested block declarations work inside one-line bodies.
    if let Some(bp) = find_char(chars, *cursor, body_end, '{') {
        parse_block_declaration(chars, cursor, block, *cursor, body_end, bp);
        return;
    }
    if let Some(eq) = find_char(chars, *cursor, body_end, '=') {
        parse_named_param(chars, block, *cursor, body_end, eq);
    } else {
        let value: String = chars[*cursor..body_end].iter().collect();
        let value = value.trim();
        if !value.is_empty() {
            block.push_entry(Entry::Par {
                name: String::new(),
                value: value.to_string(),
            });
        }
    }
    *cursor = body_end;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_params() {
        let bp = BlockPar::parse("AlphaBlend=1\nAlphaTest=0\n");
        assert_eq!(bp.par_get_ne("AlphaBlend"), Some("1"));
        assert_eq!(bp.par_get_ne("AlphaTest"), Some("0"));
        assert_eq!(bp.par_get_ne("Missing"), None);
    }

    #[test]
    fn spaces_and_comments() {
        let bp = BlockPar::parse(
            "  AlphaBlend  =   1   // set via override\n\
             // leading comment\n\
             AlphaTest  =  1\n",
        );
        assert_eq!(bp.par_get_ne("AlphaBlend"), Some("1"));
        assert_eq!(bp.par_get_ne("AlphaTest"), Some("1"));
    }

    #[test]
    fn anonymous_values() {
        let bp = BlockPar::parse("firstvalue\nsecond\n");
        let values: Vec<&str> = bp
            .entries()
            .iter()
            .filter_map(|e| match e {
                Entry::Par { name, value } if name.is_empty() => Some(value.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(values, vec!["firstvalue", "second"]);
    }

    #[test]
    fn nested_blocks() {
        let bp = BlockPar::parse(
            "Outer {\n\
             \x20 Inner = 42\n\
             \x20 Sub {\n\
             \x20   Nested = deep\n\
             \x20 }\n\
             }\n\
             Top = ok\n",
        );
        assert_eq!(bp.par_get_ne("Top"), Some("ok"));
        let outer = bp.block_get("Outer").expect("Outer block");
        assert_eq!(outer.par_get_ne("Inner"), Some("42"));
        let sub = outer.block_get("Sub").expect("Sub block");
        assert_eq!(sub.par_get_ne("Nested"), Some("deep"));
    }

    #[test]
    fn inline_empty_block() {
        let bp = BlockPar::parse("Empty {}\nAfter=1\n");
        assert!(bp.block_get("Empty").is_some());
        assert_eq!(bp.par_get_ne("After"), Some("1"));
    }

    #[test]
    fn sort_options() {
        let bp = BlockPar::parse("Sorted +sort {\n K=1\n}\nUnsorted -sort {\n K=2\n}\n");
        assert_eq!(bp.block_get("Sorted").unwrap().par_get_ne("K"), Some("1"));
        assert_eq!(bp.block_get("Unsorted").unwrap().par_get_ne("K"), Some("2"));
    }

    #[test]
    fn from_file_option() {
        let bp = BlockPar::parse("Child =path/to/file.txt {\n}\n");
        let from_file = bp.entries().iter().find_map(|e| match e {
            Entry::Block { from_file, .. } => from_file.clone(),
            _ => None,
        });
        assert_eq!(from_file.as_deref(), Some("path/to/file.txt"));
    }

    #[test]
    fn utf16_bom() {
        let mut raw = vec![0xFFu8, 0xFE];
        for ch in "AlphaTest=1\n".encode_utf16() {
            raw.extend_from_slice(&ch.to_le_bytes());
        }
        let bp = BlockPar::parse_bytes(&raw);
        assert_eq!(bp.par_get_ne("AlphaTest"), Some("1"));
    }

    #[test]
    fn case_insensitive_lookup() {
        let bp = BlockPar::parse("alphatest = 1\n");
        assert_eq!(bp.par_get_ne("AlphaTest"), Some("1"));
    }

    #[test]
    fn par_skips_block_with_same_name() {
        let bp = BlockPar::parse(
            "Dup {\n\
             \x20 Inner = 1\n\
             }\n\
             Dup = value\n",
        );
        assert_eq!(bp.par_get_ne("Dup"), Some("value"));
        assert!(bp.block_get("Dup").is_some());
    }

    #[test]
    fn crlf_line_endings() {
        let bp = BlockPar::parse("AlphaBlend=1\r\nAlphaTest=0\r\n");
        assert_eq!(bp.par_get_ne("AlphaBlend"), Some("1"));
        assert_eq!(bp.par_get_ne("AlphaTest"), Some("0"));
    }

    #[test]
    fn resolve_includes_splices_file_contents() {
        let mut bp = BlockPar::parse("Shared =other.bpt {\n}\n");
        bp.resolve_includes(&mut |path| {
            assert_eq!(path, "other.bpt");
            Some(b"FromFile = yes\n".to_vec())
        });
        assert_eq!(
            bp.block_get("Shared").unwrap().par_get_ne("FromFile"),
            Some("yes")
        );
    }

    #[test]
    fn resolve_includes_recurses_into_loaded_files() {
        let mut bp = BlockPar::parse("Outer =first.bpt {\n}\n");
        bp.resolve_includes(&mut |path| match path {
            "first.bpt" => Some(b"Inner =second.bpt {\n}\n".to_vec()),
            "second.bpt" => Some(b"Deep = done\n".to_vec()),
            _ => None,
        });
        let deep = bp
            .block_get("Outer")
            .and_then(|o| o.block_get("Inner"))
            .and_then(|i| i.par_get_ne("Deep"));
        assert_eq!(deep, Some("done"));
    }
}
