//! Shared constants and utilities — ports Common.hpp.

pub const TEX_BOTTOM_SIZE: usize = 64;
pub const MAP_GROUP_SIZE: i32 = 10;
pub const WATER_LEVEL: f32 = -2.0;
pub const WATER_SIZE: usize = 16;
pub const WATER_ALPHA_SIZE: usize = 64;
pub const WATER_TEXTURE_SCALE: f32 = 1.0 / 16.0;

/// Linear fog parameters (MatrixMap.cpp:2213-2214 + MatrixCamera.hpp:58-59).
/// MAX_VIEW_DISTANCE=4000, FOG_NEAR_K=0.5, FOG_FAR_K=0.7.
pub const FOG_START: f32 = 4000.0 * 0.5;
pub const FOG_END: f32 = 4000.0 * 0.7;

/// Unpack a u32 RGB color (0x00RRGGBB) into `[r, g, b]` floats in [0, 1].
pub fn unpack_rgb(c: u32) -> [f32; 3] {
    [
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
    ]
}

pub const CELLFLAG_LAND: u8 = 1 << 0;
pub const CELLFLAG_WATER: u8 = 1 << 1;
pub const CELLFLAG_BRIDGE: u8 = 1 << 2;
pub const CELLFLAG_INSHORE: u8 = 1 << 3;
pub const CELLFLAG_FLAT: u8 = 1 << 4;
pub const CELLFLAG_DOWN: u8 = 1 << 5;

// ── TRACE_* masks (MatrixMap.hpp:29-46) ──────────────────────────────────
//
// Bitmask used by `FindObjects` / `Trace`. Callers OR together the
// object-type bits (+ optional flags like `TRACE_SKIP_INVISIBLE`) to
// filter candidates.

pub const TRACE_LANDSCAPE:       u32 = 1 << 0;
pub const TRACE_WATER:           u32 = 1 << 1;
pub const TRACE_ROBOT:           u32 = 1 << 2;
pub const TRACE_BUILDING:        u32 = 1 << 3;
pub const TRACE_OBJECT:          u32 = 1 << 4;
pub const TRACE_CANNON:          u32 = 1 << 5;
pub const TRACE_FLYER:           u32 = 1 << 6;
pub const TRACE_ANYOBJECT:       u32 = TRACE_OBJECT | TRACE_BUILDING | TRACE_ROBOT | TRACE_CANNON | TRACE_FLYER;
pub const TRACE_OBJECTSPHERE:    u32 = 1 << 10;
pub const TRACE_SKIP_INVISIBLE:  u32 = 1 << 11;
pub const TRACE_NONOBJECT:       u32 = TRACE_LANDSCAPE | TRACE_WATER;

// ── EObjectTypeProperty (Common.hpp:176-191) ─────────────────────────────
//
// Index into the `*`-separated Ids row. `m_Ids[type_id]` is a string
// like "palm\*palm.vo\*palm.png\*\*\*\*\*Burn,Tex,Burn01\*\*0.05\*0"
// where each `*`-delimited field corresponds to one of these properties.
// Used by `CMatrixMapObject::Init` and friends.
pub const OTP_PATH:           usize = 0;
pub const OTP_VO:             usize = 1;
pub const OTP_TEXTURE:        usize = 2;
pub const OTP_TEXTURE_GLOSS:  usize = 3;
pub const OTP_TEXTURE_BACK:   usize = 4;
pub const OTP_TEXTURE_MASK:   usize = 5;
pub const OTP_TEXTURE_SCROLL: usize = 6;
pub const OTP_SHADOW:         usize = 7;
pub const OTP_BEHAVIOUR:      usize = 8;
pub const OTP_BIAS:           usize = 9;
pub const OTP_INVLOGIC:       usize = 10;

/// Side IDs — MatrixSide.hpp:16. Side 0 is neutral/world-owned; side 1
/// is the player; higher numbers are AI-controlled factions.
pub const PLAYER_SIDE: i32 = 1;

/// Ablaze tick period (MatrixMapStatic.hpp:93): the CMatrixMapObject
/// ablaze logic emits a fire/smoke effect every 90ms of sim time.
pub const OBJECT_ABLAZE_PERIOD_MS: i32 = 90;

/// Total ablaze time before an object is marked BURNED
/// (MatrixObject.cpp:1580). After 5 seconds of being ablaze, the
/// object transitions to the burned-out state and loses the ablaze
/// flag on its next TTL expiry.
pub const OBJECT_ABLAZE_BURNED_AT_MS: i32 = 5000;

/// Hit-point bar linger time after a hit (MatrixMapStatic.hpp:98).
/// A damaged object shows its hp bar for 1 second after each hit —
/// resets on every subsequent damage call.
pub const HITPOINT_SHOW_TIME_MS: i32 = 1000;

/// Base-platform open/close speed (MatrixObjectBuilding.hpp:13). In
/// units of `BaseFloor` per ms. `m_BaseFloor` ranges 0..1; speed
/// 0.0008/ms → ~1.25 seconds for a full open or close.
pub const BASE_FLOOR_SPEED: f32 = 0.0008;

// ── Binary read helpers ─────────────────────────────────────────────────────

pub fn rd_u32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}
pub fn rd_i32(d: &[u8], o: &mut usize) -> i32 {
    let v = i32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}
pub fn rd_f32(d: &[u8], o: &mut usize) -> f32 {
    let v = f32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}
pub fn rd_u16(d: &[u8], o: &mut usize) -> u16 {
    let v = u16::from_le_bytes([d[*o], d[*o + 1]]);
    *o += 2;
    v
}
