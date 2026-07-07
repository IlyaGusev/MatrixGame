//! Port of `CMatrixCursor` (MatrixCursor.cpp) — the in-engine software
//! cursor. The original ships five modes defined by the `Cursors`
//! block in robots.dat (`<texture>?<hotx>,<hoty>,<frames>,<size>,<period>`,
//! e.g. `arrow = Matrix\Textures\Cursors\arrow_blue?6,2,-16,32,50`);
//! each texture is a grid of `size`-px frames animated every `period`
//! ms. A negative frame count selects ping-pong animation
//! (CURSOR_REVERSEANIM). We always take the software path
//! (`g_Config.m_SoftwareCursor`) — the hardware `HICON` path has no
//! wgpu/winit equivalent.

use crate::matrix_lib::base::storage::Storage;

// StringConstants.hpp:664-668.
pub const CURSOR_ARROW: &str = "arrow";
pub const CURSOR_CROSS_BLUE: &str = "cross_blue";
pub const CURSOR_CROSS_RED: &str = "cross_red";
pub const CURSOR_CROSS_YELLOW: &str = "cross_yellow";
pub const CURSOR_STAR: &str = "star";

const CURSOR_NAMES: [&str; 5] = [
    CURSOR_ARROW,
    CURSOR_CROSS_BLUE,
    CURSOR_CROSS_RED,
    CURSOR_CROSS_YELLOW,
    CURSOR_STAR,
];

struct CursorDef {
    name: &'static str,
    tex_path: String,
    hot_x: i32,
    hot_y: i32,
    /// `m_FramesCnt` — already absolute; `reverse` carries the sign.
    frames_cnt: i32,
    reverse: bool,
    size: i32,
    period: i32,
}

/// Screen-space draw request for the current frame, consumed by the
/// interface renderer (the `CInstDraw` quad of `CMatrixCursor::Draw`).
pub struct CursorSprite {
    pub tex_path: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// `[u0, v0, u1, v1]` — half-texel-inset frame rect (CalcUV).
    pub uv: [f32; 4],
}

pub struct MatrixCursor {
    defs: Vec<CursorDef>,
    cur: Option<usize>,
    frame: i32,
    frame_inc: i32,
    /// `m_Time` / `m_NextCursorTime` — running counters that survive
    /// `Select` (only `m_Frame` resets), so the animation cadence
    /// carries across cursor switches like the original.
    time: i32,
    next_time: i32,
    pub pos: [f32; 2],
}

impl MatrixCursor {
    /// Parse the `Cursors` block (MatrixConfig.cpp:358-368). Missing
    /// entries are skipped — `select` on them is then a no-op.
    pub fn new(stor: &Storage) -> Self {
        let mut defs = Vec::new();
        if let Some(rec) = stor.block_record("da", "Cursors") {
            for name in CURSOR_NAMES {
                let Some(val) = stor.block_param(&rec, name) else {
                    continue;
                };
                let (tex_path, pars) = match val.split_once('?') {
                    Some((t, p)) => (t.to_string(), p),
                    None => continue,
                };
                let n: Vec<i32> = pars
                    .split(',')
                    .map(|s| s.trim().parse().unwrap_or(0))
                    .collect();
                if n.len() < 5 {
                    continue;
                }
                defs.push(CursorDef {
                    name,
                    tex_path,
                    hot_x: n[0],
                    hot_y: n[1],
                    frames_cnt: n[2].abs().max(1),
                    reverse: n[2] < 0,
                    size: n[3],
                    period: n[4],
                });
            }
        }
        if defs.is_empty() {
            log::warn!("cursor: no Cursors block in robots.dat");
        }
        Self {
            defs,
            cur: None,
            frame: 0,
            frame_inc: 1,
            time: 0,
            next_time: 0,
            pos: [-1.0, -1.0],
        }
    }

    /// Texture paths for atlas preloading.
    pub fn tex_paths(&self) -> impl Iterator<Item = &str> {
        self.defs.iter().map(|d| d.tex_path.as_str())
    }

    /// `CMatrixCursor::Select` — same-cursor calls return early, a
    /// switch restarts at frame 0 with forward step.
    pub fn select(&mut self, name: &str) {
        if self.cur.map(|i| self.defs[i].name) == Some(name) {
            return;
        }
        let Some(idx) = self.defs.iter().position(|d| d.name == name) else {
            log::warn!("cursor: unknown cursor {name}");
            return;
        };
        self.cur = Some(idx);
        self.frame = 0;
        self.frame_inc = 1;
    }

    /// `CMatrixCursor::Takt` — frame stepping, ping-pong when reversed.
    pub fn takt(&mut self, ms: i32) {
        self.time += ms;
        let Some(d) = self.cur.map(|i| &self.defs[i]) else {
            return;
        };
        if d.frames_cnt < 2 || d.period <= 0 {
            return;
        }
        while self.time > self.next_time {
            self.next_time += d.period;
            if !d.reverse {
                self.frame += 1;
                if self.frame >= d.frames_cnt {
                    self.frame = 0;
                }
            } else {
                self.frame += self.frame_inc;
                if self.frame_inc > 0 {
                    if self.frame >= d.frames_cnt {
                        self.frame -= 2;
                        self.frame_inc = -self.frame_inc;
                    }
                } else if self.frame < 0 {
                    self.frame += 2;
                    self.frame_inc = -self.frame_inc;
                }
            }
        }
    }

    /// `Draw` geometry + `CalcUV` for the current frame. `tex_w/tex_h`
    /// come from the loaded atlas; `scale` is the port's UI scale
    /// (the original draws at a fixed 32 px because it targets the
    /// 1024×768 design resolution this port scales from).
    pub fn sprite(&self, tex_w: u32, tex_h: u32, scale: f32) -> Option<CursorSprite> {
        let d = self.cur.map(|i| &self.defs[i])?;
        if self.pos[0] < 0.0 || tex_w == 0 || tex_h == 0 {
            return None;
        }
        let inv_x = 1.0 / tex_w as f32;
        let inv_y = 1.0 / tex_h as f32;
        // CalcUV (MatrixCursor.cpp:164-175) — row from frame*size/tex_w,
        // column from the per-row frame count.
        let in_line = (tex_w as i32 / d.size).max(1);
        let y = ((self.frame * d.size) as f32 * inv_x) as i32;
        let x = self.frame - y * in_line;
        let uv = [
            (x * d.size) as f32 * inv_x + inv_x * 0.5,
            (y * d.size) as f32 * inv_y + inv_y * 0.5,
            ((x + 1) * d.size) as f32 * inv_x - inv_x * 0.5,
            ((y + 1) * d.size) as f32 * inv_y - inv_y * 0.5,
        ];
        Some(CursorSprite {
            tex_path: d.tex_path.clone(),
            x: self.pos[0] - d.hot_x as f32 * scale,
            y: self.pos[1] - d.hot_y as f32 * scale,
            w: d.size as f32 * scale,
            h: d.size as f32 * scale,
            uv,
        })
    }

    pub fn tex_path(&self) -> Option<&str> {
        self.cur.map(|i| self.defs[i].tex_path.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cursor_with(frames: i32, period: i32) -> MatrixCursor {
        MatrixCursor {
            defs: vec![CursorDef {
                name: CURSOR_ARROW,
                tex_path: "t".into(),
                hot_x: 6,
                hot_y: 2,
                frames_cnt: frames.abs(),
                reverse: frames < 0,
                size: 32,
                period,
            }],
            cur: Some(0),
            frame: 0,
            frame_inc: 1,
            time: 0,
            next_time: 0,
            pos: [100.0, 100.0],
        }
    }

    #[test]
    fn forward_anim_wraps() {
        let mut c = cursor_with(4, 50);
        let mut seen = Vec::new();
        for _ in 0..6 {
            c.takt(50);
            seen.push(c.frame);
        }
        assert_eq!(seen, vec![1, 2, 3, 0, 1, 2]);
    }

    #[test]
    fn reverse_anim_ping_pongs() {
        let mut c = cursor_with(-4, 50);
        let mut seen = Vec::new();
        for _ in 0..8 {
            c.takt(50);
            seen.push(c.frame);
        }
        // 0→1→2→3→2→1→0→1→2 (bounce at both ends, no double-hold).
        assert_eq!(seen, vec![1, 2, 3, 2, 1, 0, 1, 2]);
    }

    #[test]
    fn calc_uv_walks_4x4_grid() {
        let mut c = cursor_with(16, 50);
        c.frame = 5; // row 1, col 1 of a 128×128 4×4 grid
        let s = c.sprite(128, 128, 1.0).unwrap();
        let e = 1.0 / 128.0;
        assert!((s.uv[0] - (32.0 / 128.0 + e * 0.5)).abs() < 1e-6);
        assert!((s.uv[1] - (32.0 / 128.0 + e * 0.5)).abs() < 1e-6);
        assert!((s.uv[2] - (64.0 / 128.0 - e * 0.5)).abs() < 1e-6);
        assert!((s.uv[3] - (64.0 / 128.0 - e * 0.5)).abs() < 1e-6);
        // Hotspot offset.
        assert_eq!(s.x, 100.0 - 6.0);
        assert_eq!(s.y, 100.0 - 2.0);
        assert_eq!(s.w, 32.0);
    }
}
