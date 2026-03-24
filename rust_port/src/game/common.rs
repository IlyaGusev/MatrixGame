//! Shared constants and utilities — ports Common.hpp.

pub const TEX_BOTTOM_SIZE: usize = 64;
pub const MAP_GROUP_SIZE: i32 = 10;
pub const WATER_LEVEL: f32 = -2.0;
pub const WATER_SIZE: usize = 16;
pub const WATER_ALPHA_SIZE: usize = 64;
pub const WATER_TEXTURE_SCALE: f32 = 1.0 / 16.0;

pub const CELLFLAG_LAND: u8 = 1 << 0;
pub const CELLFLAG_WATER: u8 = 1 << 1;
pub const CELLFLAG_BRIDGE: u8 = 1 << 2;
pub const CELLFLAG_INSHORE: u8 = 1 << 3;
pub const CELLFLAG_FLAT: u8 = 1 << 4;
pub const CELLFLAG_DOWN: u8 = 1 << 5;

// ── Binary read helpers ─────────────────────────────────────────────────────

pub fn rd_u32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]);
    *o += 4; v
}
pub fn rd_i32(d: &[u8], o: &mut usize) -> i32 {
    let v = i32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]);
    *o += 4; v
}
pub fn rd_f32(d: &[u8], o: &mut usize) -> f32 {
    let v = f32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]);
    *o += 4; v
}
pub fn rd_u16(d: &[u8], o: &mut usize) -> u16 {
    let v = u16::from_le_bytes([d[*o], d[*o+1]]);
    *o += 2; v
}
