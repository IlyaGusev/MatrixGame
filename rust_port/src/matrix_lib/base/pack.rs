//! Reader for SR2 .pkg archive format.
//!
//! Format:
//!   - Bytes 0..4: u32 LE root folder offset
//!   - At root offset: SFolderRec (12 bytes) followed by SFileRec[] array
//!   - Folders recurse: SFileRec.m_Offset points to another SFolderRec
//!   - Compressed files use ZL03 blocks (zlib, 64KB logical blocks)

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Cursor, Read};

const MAX_FILENAME_LENGTH: usize = 63;
const FILE_REC_SIZE: usize = 158; // under /Zp1 (1-byte packing)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Text = 0,
    Binary = 1,
    Compressed = 2,
    Folder = 3,
}

impl FileType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::Text),
            1 => Ok(Self::Binary),
            2 => Ok(Self::Compressed),
            3 => Ok(Self::Folder),
            _ => bail!("unknown file type: {v}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub name: String,
    pub real_name: String,
    pub file_type: FileType,
    pub size: u32,
    pub real_size: u32,
    pub offset: u32,
}

/// Parsed .pkg archive — holds the raw data and a path→entry index.
pub struct PkgArchive {
    data: Vec<u8>,
    entries: HashMap<String, FileEntry>,
}

impl PkgArchive {
    /// Parse a .pkg archive from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let root_offset = read_u32(&data, 0)? as usize;
        let mut entries = HashMap::new();
        parse_folder(&data, root_offset, "", &mut entries)?;

        log::info!("pkg: loaded {} entries", entries.len());
        Ok(Self { data, entries })
    }

    /// List all file paths in the archive.
    pub fn list_files(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a path exists.
    pub fn exists(&self, path: &str) -> bool {
        self.entries.contains_key(&normalize_path(path))
    }

    /// Read a file's contents, decompressing if needed.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let key = normalize_path(path);
        let entry = self
            .entries
            .get(&key)
            .with_context(|| format!("file not found in pkg: {path}"))?;

        match entry.file_type {
            FileType::Text | FileType::Binary => self.read_raw(entry),
            FileType::Compressed => self.read_compressed(entry),
            FileType::Folder => bail!("cannot read a folder: {path}"),
        }
    }

    fn read_raw(&self, entry: &FileEntry) -> Result<Vec<u8>> {
        // Data starts at offset+4 (skip 4-byte block size header)
        let start = entry.offset as usize + 4;
        let end = start + entry.real_size as usize;
        if end > self.data.len() {
            bail!(
                "raw file data out of bounds: {}..{} (pkg size {})",
                start,
                end,
                self.data.len()
            );
        }
        Ok(self.data[start..end].to_vec())
    }

    fn read_compressed(&self, entry: &FileEntry) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(entry.real_size as usize);
        // Skip 4-byte size header at m_Offset, same as raw files
        let mut block_offset = entry.offset as usize + 4;

        while result.len() < entry.real_size as usize {
            if block_offset + 4 > self.data.len() {
                bail!("compressed block header out of bounds");
            }
            let block_size = read_u32(&self.data, block_offset)? as usize;
            block_offset += 4;

            if block_offset + block_size > self.data.len() {
                bail!("compressed block data out of bounds");
            }

            let block_data = &self.data[block_offset..block_offset + block_size];
            block_offset += block_size;

            // ZL03 format: 4-byte magic "ZL03", 4-byte uncompressed size, then zlib data
            if block_data.len() < 8 {
                bail!("compressed block too small for ZL03 header");
            }
            if &block_data[0..4] != b"ZL03" && &block_data[0..4] != b"ZL02" {
                bail!("expected ZL02/ZL03 magic, got {:?}", &block_data[0..4]);
            }
            let _uncompressed_size =
                u32::from_le_bytes([block_data[4], block_data[5], block_data[6], block_data[7]]);

            let mut decoder = flate2::read::ZlibDecoder::new(Cursor::new(&block_data[8..]));
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .context("zlib decompression failed")?;

            result.extend_from_slice(&decompressed);
        }

        result.truncate(entry.real_size as usize);
        Ok(result)
    }
}

fn parse_folder(
    data: &[u8],
    offset: usize,
    prefix: &str,
    entries: &mut HashMap<String, FileEntry>,
) -> Result<()> {
    if offset + 12 > data.len() {
        bail!("folder record out of bounds at offset {offset}");
    }

    let _folder_size = read_u32(data, offset)?;
    let rec_num = read_u32(data, offset + 4)? as usize;
    let rec_size = read_u32(data, offset + 8)? as usize;

    if rec_size < FILE_REC_SIZE {
        bail!("unexpected record size: {rec_size} (expected >= {FILE_REC_SIZE})");
    }

    let records_start = offset + 12;

    for i in 0..rec_num {
        let rec_offset = records_start + i * rec_size;
        if rec_offset + rec_size > data.len() {
            bail!("file record {i} out of bounds");
        }

        let entry = parse_file_rec(data, rec_offset)?;

        // Skip free records (m_Free is at offset 142 within record)
        let free = read_u32(data, rec_offset + 142)?;
        if free != 0 {
            continue;
        }

        if entry.file_type == FileType::Folder {
            let folder_path = if prefix.is_empty() {
                entry.real_name.clone()
            } else {
                format!("{}/{}", prefix, entry.real_name)
            };
            parse_folder(data, entry.offset as usize, &folder_path, entries)?;
        } else {
            let full_path = if prefix.is_empty() {
                entry.real_name.clone()
            } else {
                format!("{}/{}", prefix, entry.real_name)
            };
            let key = normalize_path(&full_path);
            entries.insert(key, entry);
        }
    }
    Ok(())
}

fn parse_file_rec(data: &[u8], offset: usize) -> Result<FileEntry> {
    let size = read_u32(data, offset)?;
    let real_size = read_u32(data, offset + 4)?;
    let name = read_fixed_string(data, offset + 8, MAX_FILENAME_LENGTH);
    let real_name = read_fixed_string(data, offset + 8 + MAX_FILENAME_LENGTH, MAX_FILENAME_LENGTH);
    let file_type = FileType::from_u32(read_u32(data, offset + 8 + 63 + 63)?)?;
    let file_offset = read_u32(data, offset + 8 + 63 + 63 + 4 + 4 + 4 + 4)?; // skip m_Type, m_NType, m_Free, m_Date

    Ok(FileEntry {
        name,
        real_name,
        file_type,
        size,
        real_size,
        offset: file_offset,
    })
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        bail!("read_u32 out of bounds at offset {offset}");
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_fixed_string(data: &[u8], offset: usize, max_len: usize) -> String {
    let slice = &data[offset..offset + max_len];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(max_len);
    String::from_utf8_lossy(&slice[..end]).to_string()
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").to_uppercase()
}
