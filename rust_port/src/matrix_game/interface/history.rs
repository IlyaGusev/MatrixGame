//! Port of `CHistory` (Interface/CHistory.{cpp,h}).
//!
//! The C++ keeps a global `g_ConfigHistory` (CHistory.cpp:11) holding
//! every preset the player has built; the `hisleft` / `hisright`
//! buttons cycle through it via `CHistory::PrevConfig` / `NextConfig`,
//! and `RemoteBuild` calls `AddConfig` to push a new entry. We mirror
//! the global by parking the instance on `IFaceList` (the rust analog
//! of the C++ UI singleton container).

use crate::matrix_game::robot_units::{RobotConfig, MAX_WEAPON_CNT};

/// Port of `CHistory` (CHistory.h:8-28). Keeps the doubly-linked list
/// of committed configs as a `Vec` (insertion is always at the tail
/// and the cursor is an index, so a linked list would buy nothing).
#[derive(Debug, Clone, Default)]
pub struct ConfigHistory {
    /// Configs in oldest→newest order. `LIST_ADD` (CHistory.cpp:42)
    /// appends to the tail.
    pub configs: Vec<RobotConfig>,
    /// Pointer-equivalent into `configs`. -1 when empty. Port of
    /// `m_CurrentConfig` (CHistory.h:9).
    pub cursor: i32,
}

impl ConfigHistory {
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            cursor: -1,
        }
    }

    /// Port of `CHistory::AddConfig` (CHistory.cpp:35-43). First removes
    /// any existing entry whose head, chassis, hull and all weapons match
    /// the new one (`DelConfig`, CHistory.cpp:45-73), then appends the
    /// new entry and parks the cursor on it.
    pub fn add(&mut self, cfg: RobotConfig) {
        self.configs.retain(|existing| {
            if existing.head.kind != cfg.head.kind
                || existing.chassis.kind != cfg.chassis.kind
                || existing.hull.unit.kind != cfg.hull.unit.kind
            {
                return true;
            }
            for i in 0..MAX_WEAPON_CNT {
                if existing.weapon[i].kind != cfg.weapon[i].kind {
                    return true;
                }
            }
            false
        });
        self.configs.push(cfg);
        self.cursor = self.configs.len() as i32 - 1;
    }

    /// Port of `CHistory::PrevConfig` (CHistory.cpp:90-97). Walks the
    /// cursor one slot back when `m_PrevConfig` is non-null. Returns
    /// the new cursor's config so the caller can apply it via
    /// `RobotBuilder::apply_config` (mirrors `LoadCurrentConfigToConstructor`).
    pub fn prev(&mut self) -> Option<RobotConfig> {
        if self.cursor <= 0 {
            return None;
        }
        self.cursor -= 1;
        Some(self.configs[self.cursor as usize])
    }

    /// Port of `CHistory::NextConfig` (CHistory.cpp:99-106).
    pub fn next(&mut self) -> Option<RobotConfig> {
        if self.cursor < 0 || (self.cursor as usize) + 1 >= self.configs.len() {
            return None;
        }
        self.cursor += 1;
        Some(self.configs[self.cursor as usize])
    }

    /// Port of `CHistory::IsPrev` (CHistory.h:24).
    pub fn is_prev(&self) -> bool {
        self.cursor > 0
    }

    /// Port of `CHistory::IsNext` (CHistory.h:23).
    pub fn is_next(&self) -> bool {
        self.cursor >= 0 && (self.cursor as usize) + 1 < self.configs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::robot_units::RobotUnitKind;

    #[test]
    fn add_is_unbounded() {
        let mut h = ConfigHistory::new();
        for i in 0..200 {
            let mut c = RobotConfig::new();
            c.chassis.kind = RobotUnitKind((i + 1) as i32);
            h.add(c);
        }
        assert_eq!(h.configs.len(), 200);
    }

    #[test]
    fn add_deduplicates_matching_config() {
        let mut h = ConfigHistory::new();
        let mut c = RobotConfig::new();
        c.chassis.kind = RobotUnitKind(1);
        c.hull.unit.kind = RobotUnitKind(2);
        c.head.kind = RobotUnitKind(3);
        h.add(c);
        h.add(c);
        h.add(c);
        assert_eq!(h.configs.len(), 1);
        assert_eq!(h.cursor, 0);

        let mut other = RobotConfig::new();
        other.chassis.kind = RobotUnitKind(99);
        h.add(other);
        h.add(c);
        assert_eq!(h.configs.len(), 2);
        assert_eq!(h.configs[0].chassis.kind, RobotUnitKind(99));
        assert_eq!(h.configs[1].chassis.kind, RobotUnitKind(1));
        assert_eq!(h.cursor, 1);
    }

    #[test]
    fn prev_next_walks_cursor() {
        let mut h = ConfigHistory::new();
        for i in 1..=3 {
            let mut c = RobotConfig::new();
            c.chassis.kind = RobotUnitKind(i);
            h.add(c);
        }
        assert_eq!(h.cursor, 2);
        assert!(h.is_prev());
        assert!(!h.is_next());
        let p = h.prev().unwrap();
        assert_eq!(p.chassis.kind, RobotUnitKind(2));
        assert!(h.is_prev());
        assert!(h.is_next());
        let p = h.prev().unwrap();
        assert_eq!(p.chassis.kind, RobotUnitKind(1));
        assert!(!h.is_prev());
        assert!(h.prev().is_none());
        let n = h.next().unwrap();
        assert_eq!(n.chassis.kind, RobotUnitKind(2));
    }
}
