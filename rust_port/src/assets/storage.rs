//! Parser for SR2 CStorage binary format (STRG files).
//!
//! Used by .CMAP map files and other game data.
//!
//! Format:
//!   STRG magic (4 bytes) + version (u32) + [compressed payload | records]
//!   Each record: WStr name, u32 item_count, items...
//!   Each item: WStr name, u32 type, u32 data_size, raw data (CDataBuf binary)

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

const STRG_MAGIC: u32 = 0x47525453; // "STRG" little-endian
const ST_COMPRESSED: u32 = 1 << 31;

/// Parsed CStorage — a collection of named records, each with named data columns.
pub struct Storage {
    /// Key: (record_name, item_name), Value: raw CDataBuf bytes
    items: HashMap<(String, String), DataBuf>,
}

/// Parsed CDataBuf — an array-of-arrays with typed elements.
pub struct DataBuf {
    data: Vec<u8>,
    element_size: usize,
    arrays: Vec<ArrayEntry>,
}

struct ArrayEntry {
    offset: usize,
    count: usize,
}

impl DataBuf {
    fn parse(raw: &[u8]) -> Result<Self> {
        if raw.len() < 12 {
            bail!("DataBuf too small: {} bytes", raw.len());
        }
        let alloc_table_disp = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
        let arrays_count = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
        let element_size = i32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;

        let mut arrays = Vec::with_capacity(arrays_count);
        for i in 0..arrays_count {
            let te_offset = alloc_table_disp + i * 12;
            if te_offset + 12 > raw.len() {
                bail!("DataBuf table entry out of bounds");
            }
            let disp = u32::from_le_bytes([
                raw[te_offset],
                raw[te_offset + 1],
                raw[te_offset + 2],
                raw[te_offset + 3],
            ]) as usize;
            let count = u32::from_le_bytes([
                raw[te_offset + 4],
                raw[te_offset + 5],
                raw[te_offset + 6],
                raw[te_offset + 7],
            ]) as usize;
            arrays.push(ArrayEntry {
                offset: disp,
                count,
            });
        }

        Ok(Self {
            data: raw.to_vec(),
            element_size,
            arrays,
        })
    }

    /// Number of arrays (rows) in this buffer.
    pub fn arrays_count(&self) -> usize {
        self.arrays.len()
    }

    /// Get raw bytes for array `i`.
    pub fn get_bytes(&self, i: usize) -> &[u8] {
        let entry = &self.arrays[i];
        let start = entry.offset;
        let end = start + entry.count * self.element_size;
        &self.data[start..end]
    }

    /// Get array `i` as a UTF-16LE string (for ST_WCHAR buffers).
    pub fn get_as_wstr(&self, i: usize) -> String {
        let entry = &self.arrays[i];
        let start = entry.offset;
        let byte_len = entry.count * 2; // element_size should be 2 for wchar
        let slice = &self.data[start..start + byte_len];
        let chars: Vec<u16> = slice
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&chars)
    }

    /// Find index of array matching a UTF-16LE string (for ST_WCHAR buffers).
    pub fn find_as_wstr(&self, val: &str) -> Option<usize> {
        for i in 0..self.arrays.len() {
            if self.get_as_wstr(i) == val {
                return Some(i);
            }
        }
        None
    }
}

impl Storage {
    /// Parse a CStorage from raw STRG bytes (e.g. a .CMAP file).
    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        let mut pos = 0;

        let magic = read_u32(raw, &mut pos)?;
        if magic != STRG_MAGIC {
            bail!("not a STRG file (magic: 0x{:08x})", magic);
        }
        let version = read_u32(raw, &mut pos)?;
        if version > 1 {
            bail!("unsupported STRG version: {version}");
        }

        // If version 1, decompress the rest
        let data;
        let buf = if version == 1 {
            data = zl_decompress_all(&raw[pos..])?;
            &data[..]
        } else {
            raw
        };
        // For version 0, continue reading from current position
        // For version 1, read from decompressed buffer (starts at record_count)
        let mut rpos = if version == 0 { pos } else { 0 };

        let record_count = read_u32(buf, &mut rpos)? as usize;

        let mut items = HashMap::new();

        for _ in 0..record_count {
            let rec_name = read_wstr(buf, &mut rpos)?;
            let item_count = read_u32(buf, &mut rpos)? as usize;

            for _ in 0..item_count {
                let item_name = read_wstr(buf, &mut rpos)?;
                let mut item_type = read_u32(buf, &mut rpos)?;
                let data_size = read_u32(buf, &mut rpos)? as usize;

                if rpos + data_size > buf.len() {
                    bail!(
                        "item data out of bounds: need {} bytes at offset {}, buf len {}",
                        data_size,
                        rpos,
                        buf.len()
                    );
                }

                let item_data = if item_type & ST_COMPRESSED != 0 {
                    item_type &= !ST_COMPRESSED;
                    zl_decompress_all(&buf[rpos..rpos + data_size])?
                } else {
                    buf[rpos..rpos + data_size].to_vec()
                };
                rpos += data_size;

                let db = DataBuf::parse(&item_data)
                    .with_context(|| format!("parsing DataBuf for {rec_name}/{item_name}"))?;
                items.insert((rec_name.clone(), item_name), db);
            }
        }

        log::info!(
            "storage: loaded {} records with {} items total",
            record_count,
            items.len()
        );
        Ok(Self { items })
    }

    /// Get a DataBuf by record name and item name.
    pub fn get_buf(&self, record: &str, item: &str) -> Option<&DataBuf> {
        self.items.get(&(record.to_string(), item.to_string()))
    }

    /// Resolve a child BlockPar by name. BlockPars are serialized as
    /// `StoreBlockPar` (CStorage.cpp:481): columns "2" hold child block
    /// names, "3" the unique record name each child lives under. Returns
    /// the record name of the matching child, or `None` if no child with
    /// that name exists in `record`.
    pub fn block_record(&self, record: &str, block_name: &str) -> Option<String> {
        let names = self.get_buf(record, "2")?;
        let records = self.get_buf(record, "3")?;
        let idx = names.find_as_wstr(block_name)?;
        Some(records.get_as_wstr(idx))
    }

    /// Read a scalar BlockPar parameter. Columns "0" and "1" hold parameter
    /// keys and string values respectively (CStorage.cpp:494-502).
    pub fn block_param(&self, record: &str, key: &str) -> Option<String> {
        let keys = self.get_buf(record, "0")?;
        let values = self.get_buf(record, "1")?;
        let idx = keys.find_as_wstr(key)?;
        Some(values.get_as_wstr(idx))
    }

    /// Print all record/item keys and their array counts.
    pub fn dump_structure(&self) {
        let mut keys: Vec<_> = self.items.keys().collect();
        keys.sort();
        println!("=== Storage structure ({} items) ===", keys.len());
        for (rec, item) in &keys {
            let db = &self.items[&(rec.clone(), item.clone())];
            println!(
                "  {rec}/{item}: {} arrays, elem_size={}",
                db.arrays_count(),
                db.element_size
            );
        }
    }
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > data.len() {
        bail!("read_u32 out of bounds at {}", *pos);
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

/// Read a null-terminated UTF-16LE string from buffer, advancing position past the null.
fn read_wstr(data: &[u8], pos: &mut usize) -> Result<String> {
    let start = *pos;
    // Find null terminator (0x0000)
    let mut chars = Vec::new();
    loop {
        if *pos + 2 > data.len() {
            bail!("WStr runs past end of buffer at offset {start}");
        }
        let ch = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
        *pos += 2;
        if ch == 0 {
            break;
        }
        chars.push(ch);
    }
    Ok(String::from_utf16_lossy(&chars))
}

/// Decompress ZL03 data — faithful port of CStorage.cpp:18 ZL03_UnCompress.
///
/// Format:
///   bytes 0..4: "ZL03" magic
///   bytes 4..8: i32 cnt — number of compressed blocks
///   For each block:
///     u32 szb — compressed size
///     szb bytes of zlib-compressed data
fn zl_decompress_all(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    if data.len() < 8 {
        bail!("ZL03 data too small: {} bytes", data.len());
    }
    if &data[0..4] != b"ZL03" && &data[0..4] != b"ZL02" {
        bail!("expected ZL03/ZL02 magic, got {:?}", &data[0..4]);
    }

    let cnt = i32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mut result = Vec::new();
    let mut iptr = 8usize;

    for _ in 0..cnt {
        if iptr + 4 > data.len() {
            bail!("ZL03 block header out of bounds at offset {}", iptr);
        }
        let szb = u32::from_le_bytes([data[iptr], data[iptr + 1], data[iptr + 2], data[iptr + 3]])
            as usize;
        iptr += 4;

        if iptr + szb > data.len() {
            bail!(
                "ZL03 block data out of bounds: need {} bytes at offset {}",
                szb,
                iptr
            );
        }

        let mut decoder = ZlibDecoder::new(&data[iptr..iptr + szb]);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("ZL03 zlib decompression failed")?;
        result.extend_from_slice(&decompressed);

        iptr += szb;
    }

    Ok(result)
}
