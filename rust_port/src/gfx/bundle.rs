//! Simple asset bundle format for WASM.
//!
//! Format: u32 file_count, then for each file:
//!   u32 path_len, [path_len bytes UTF-8 path], u32 data_len, [data_len bytes]
//!
//! A bundle file may additionally be zstd-compressed as a whole
//! (the pack examples write it that way — ~3x smaller downloads);
//! `from_bytes` detects the zstd magic and inflates transparently.

use anyhow::{bail, Result};
use std::collections::HashMap;

pub struct AssetBundle {
    files: HashMap<String, Vec<u8>>,
}

impl Default for AssetBundle {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetBundle {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    pub fn add(&mut self, path: &str, data: Vec<u8>) {
        // Normalise to forward slashes on insert so callers can look up
        // by either convention (the source data uses both — `Hints/0/
        // Source` is forward-slash, `Hints/Bitmaps/*` is backslash).
        self.files.insert(path.replace('\\', "/"), data);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.files.len() as u32).to_le_bytes());
        for (path, data) in &self.files {
            let path_bytes = path.as_bytes();
            buf.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(path_bytes);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
        }
        buf
    }

    /// zstd-compress the serialized bundle (pack-time only; the C zstd
    /// encoder doesn't build on wasm32).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_zstd_bytes(&self, level: i32) -> Vec<u8> {
        zstd::encode_all(std::io::Cursor::new(self.to_bytes()), level).expect("zstd encode")
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        // Whole-file zstd frame? (magic 0xFD2FB528, little-endian)
        let inflated;
        let raw = if raw.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
            use std::io::Read;
            let mut dec = ruzstd::decoding::StreamingDecoder::new(raw)
                .map_err(|e| anyhow::anyhow!("zstd header: {e}"))?;
            let mut out = Vec::new();
            dec.read_to_end(&mut out)
                .map_err(|e| anyhow::anyhow!("zstd decode: {e}"))?;
            inflated = out;
            &inflated[..]
        } else {
            raw
        };

        let mut files = HashMap::new();
        let mut pos = 0;

        if raw.len() < 4 {
            bail!("bundle too small");
        }
        let count = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
        pos += 4;

        for _ in 0..count {
            if pos + 4 > raw.len() {
                bail!("bundle truncated at path_len");
            }
            let path_len =
                u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
            pos += 4;

            if pos + path_len > raw.len() {
                bail!("bundle truncated at path");
            }
            let path = String::from_utf8_lossy(&raw[pos..pos + path_len]).to_string();
            pos += path_len;

            if pos + 4 > raw.len() {
                bail!("bundle truncated at data_len");
            }
            let data_len =
                u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
            pos += 4;

            if pos + data_len > raw.len() {
                bail!("bundle truncated at data");
            }
            let data = raw[pos..pos + data_len].to_vec();
            pos += data_len;

            // Normalise on load so callers that look up with forward
            // slashes hit entries packed with backslashes (and vice
            // versa). `add` does the same; both paths converge.
            files.insert(path.replace('\\', "/"), data);
        }

        log::info!("bundle: loaded {} files", files.len());
        Ok(Self { files })
    }

    pub fn read_file(&self, path: &str) -> Option<&[u8]> {
        // Match the slash-normalisation applied on insert so callers
        // that hand us a backslash path (raw block-param values from
        // robots.dat) still hit existing entries.
        let normalised = path.replace('\\', "/");
        self.files
            .get(&normalised)
            .or_else(|| self.files.get(path))
            .map(|v| v.as_slice())
    }

    pub fn list_files(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.files.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn zstd_roundtrip() {
        let mut b = AssetBundle::new();
        b.add("a/b.png", vec![1, 2, 3, 4]);
        b.add("robots.dat", vec![0; 4096]);
        let packed = b.to_zstd_bytes(3);
        assert!(packed.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]));
        let back = AssetBundle::from_bytes(&packed).unwrap();
        assert_eq!(back.read_file("a/b.png"), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(back.read_file("robots.dat").map(|d| d.len()), Some(4096));
        // Uncompressed bundles still parse.
        let plain = AssetBundle::from_bytes(&b.to_bytes()).unwrap();
        assert_eq!(plain.read_file("a/b.png"), Some(&[1u8, 2, 3, 4][..]));
    }
}
