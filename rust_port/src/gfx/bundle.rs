//! Simple asset bundle format for WASM.
//!
//! Format: u32 file_count, then for each file:
//!   u32 path_len, [path_len bytes UTF-8 path], u32 data_len, [data_len bytes]

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

    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
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
