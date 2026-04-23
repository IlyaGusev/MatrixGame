//! Partial port of `CWStr` (MatrixLib/Base/src/CWStr.cpp) — only the
//! field-parser helpers that the game-code uses to pull `*`/`,`/`:`-
//! separated values out of the Ids table.
//!
//! The source strings are already decoded to Rust `String`s by
//! `DataBuf::get_as_wstr`, so we work on `&str` + a set of delimiter
//! characters — matching the C++ semantics of
//! `GetStrPar(n, delims)` which splits on *any* char in `delims`.
//!
//! Not ported (yet): the buffer / refcounting / format helpers
//! on CWStr. Those land if callers need them.

/// Count of `*`/`,`/…-separated parts in `s`. Mirrors
/// `CWStr::GetCountPar` (CWStr.cpp:548-566). An empty string returns 0;
/// otherwise the result is one more than the number of delimiter hits.
pub fn count_par(s: &str, delims: &str) -> usize {
    if s.is_empty() || delims.is_empty() {
        return 0;
    }
    let mut c = 1usize;
    for ch in s.chars() {
        if delims.contains(ch) {
            c += 1;
        }
    }
    c
}

/// Nth (0-based) part of `s` split by any char in `delims`. Mirrors
/// `CWStr::GetStrPar(int np, const wchar* ogsim)` (CWStr.cpp:619-624).
/// The C++ walks `GetSmePar` → `GetLenPar` to find byte ranges; here we
/// do the same with `char_indices`. Returns `""` when `np` is past the
/// end (the C++ would `ERROR_E`; we degrade gracefully — the only
/// callers in the port treat missing fields as absent).
pub fn str_par<'a>(s: &'a str, np: usize, delims: &str) -> &'a str {
    if s.is_empty() || delims.is_empty() {
        return "";
    }
    let mut tekpar = 0usize;
    let mut start = 0usize;
    let mut iter = s.char_indices().peekable();
    while tekpar < np {
        let next = iter.find(|&(_, c)| delims.contains(c));
        match next {
            Some((i, c)) => {
                tekpar += 1;
                start = i + c.len_utf8();
            }
            None => {
                // Fell off the end before reaching `np`.
                return "";
            }
        }
    }
    // Find end: next delim from `start`, or end-of-string.
    let tail = &s[start..];
    match tail.find(|c: char| delims.contains(c)) {
        Some(end_rel) => &tail[..end_rel],
        None => tail,
    }
}

/// Port of `CWStr::CompareFirst(str)` (CWStr.cpp:686-698) — prefix match.
/// Both strings non-empty; `prefix` must be no longer than `s`.
pub fn compare_first(s: &str, prefix: &str) -> bool {
    if s.is_empty() || prefix.is_empty() {
        return false;
    }
    s.starts_with(prefix)
}

/// Integer variant of [`str_par`]. Ports the `GetIntPar` pattern
/// (CWStr reads the sub-part then parses it as int). The C++ uses
/// wcstol-style parsing which tolerates trailing garbage; we match
/// that with `take_while(is_digit|-)`. Missing / malformed → 0,
/// same as the C++ default when the wstr is empty.
pub fn int_par(s: &str, np: usize, delims: &str) -> i32 {
    let part = str_par(s, np, delims).trim();
    let mut end = 0;
    for (i, c) in part.char_indices() {
        if (i == 0 && c == '-') || c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    part[..end].parse::<i32>().unwrap_or(0)
}

/// Float variant of [`str_par`]. Ports `GetDoublePar`. Same tolerant
/// semantics as [`int_par`].
pub fn double_par(s: &str, np: usize, delims: &str) -> f64 {
    let part = str_par(s, np, delims).trim();
    let mut end = 0;
    for (i, c) in part.char_indices() {
        if (i == 0 && c == '-') || c == '.' || c.is_ascii_digit() || c == 'e' || c == 'E' {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    part[..end].parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_par_empty_and_basic() {
        assert_eq!(count_par("", "*"), 0);
        assert_eq!(count_par("a", "*"), 1);
        assert_eq!(count_par("a*b*c", "*"), 3);
        assert_eq!(count_par("a,b*c", "*,"), 3);
    }

    #[test]
    fn str_par_pulls_nth_field() {
        let s = "palm*palm.vo*palm.png***Burn,Tex,Burn01**0.05*0";
        assert_eq!(str_par(s, 0, "*"), "palm");
        assert_eq!(str_par(s, 1, "*"), "palm.vo");
        assert_eq!(str_par(s, 2, "*"), "palm.png");
        assert_eq!(str_par(s, 3, "*"), ""); // empty fields between ** **
        assert_eq!(str_par(s, 5, "*"), "Burn,Tex,Burn01");
    }

    #[test]
    fn str_par_comma_subsplit() {
        let beh = "Burn,Tex,Burn01";
        assert_eq!(str_par(beh, 0, ","), "Burn");
        assert_eq!(str_par(beh, 1, ","), "Tex");
        assert_eq!(str_par(beh, 2, ","), "Burn01");
    }

    #[test]
    fn str_par_out_of_range_returns_empty() {
        assert_eq!(str_par("a*b", 5, "*"), "");
    }

    #[test]
    fn compare_first_is_prefix_match() {
        assert!(compare_first("Burn,Tex,Burn01", "Burn"));
        assert!(compare_first("Break,2,5000", "Break"));
        assert!(!compare_first("Break,2,5000", "Burn"));
        assert!(!compare_first("", "Burn"));
        assert!(!compare_first("Burn", ""));
    }

    #[test]
    fn int_par_parses_digits_up_to_first_nonnumeric() {
        assert_eq!(int_par("Break,2,5000,Terron", 1, ","), 2);
        assert_eq!(int_par("Break,2,5000,Terron", 2, ","), 5000);
        assert_eq!(int_par("Break,-3,5000", 1, ","), -3);
        assert_eq!(int_par("Break,abc", 1, ","), 0); // malformed → 0
        assert_eq!(int_par("", 0, ","), 0);
    }

    #[test]
    fn double_par_parses_decimal() {
        assert_eq!(double_par("Sens,120.5", 1, ","), 120.5);
        assert_eq!(double_par("A,B", 1, ","), 0.0);
    }
}
