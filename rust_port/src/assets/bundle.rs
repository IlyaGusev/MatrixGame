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
    /// Create an empty bundle for writing.
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Add a file to the bundle.
    pub fn add(&mut self, path: &str, data: Vec<u8>) {
        self.files.insert(path.to_string(), data);
    }

    /// Serialize the bundle to bytes.
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

    /// Parse a bundle from bytes.
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

            files.insert(path, data);
        }

        log::info!("bundle: loaded {} files", files.len());
        Ok(Self { files })
    }

    /// Read a file from the bundle.
    pub fn read_file(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).map(|v| v.as_slice())
    }

    /// List all file paths.
    pub fn list_files(&self) -> Vec<&str> {
        self.files.keys().map(|s| s.as_str()).collect()
    }
}
