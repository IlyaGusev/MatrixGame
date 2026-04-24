//! Port of `CIFaceCounter` (Interface/CCounter.{cpp,h}).
//!
//! The C++ instance lives at `g_IFaceList->m_RCountControl`
//! (CInterface.h:310) — a single global multiplier counter beside the
//! `cobuild` button on the constructor panel. Its job is to multiply
//! the next build click N times, where N is in `[DOWN_LIMIT, UP_LIMIT]`.
//!
//! In C++ the counter directly flips its sibling `bup`/`bdown` button
//! elements between `IFACE_NORMAL` and `IFACE_DISABLED` via stored
//! `CIFaceElement*` pointers. We don't keep element pointers; instead
//! we track `button_up_enabled` / `button_down_enabled` flags here and
//! the per-frame visibility refresh in `CInterface::refresh_base_visibility_v2`
//! reads them when computing the elements' `cur_state`.

use crate::matrix_game::config::Resource;
use crate::matrix_game::interface::constructor::UnitPrice;

/// Port of `COUNTER_IMAGES` (CCounter.h:6).
pub const COUNTER_IMAGES: usize = 7;
/// Port of `UP_LIMIT` (CCounter.h:7).
pub const UP_LIMIT: i32 = 6;
/// Port of `DOWN_LIMIT` (CCounter.h:8).
pub const DOWN_LIMIT: i32 = 1;

/// Inputs for `CheckUp` (CCounter.cpp:66-99). The C++ pulls these
/// straight off the player side + active building globals; we pass them
/// in so this module stays free of direct world coupling.
#[derive(Debug, Clone, Copy)]
pub struct CheckUpCtx {
    /// Per-resource cost of the current live preview, single robot —
    /// returned by `m_Constructor->GetConstructionPrice` (CCounter.cpp:77).
    pub per_unit_price: UnitPrice,
    /// `IsEnoughResources` predicate inputs (CCounter.cpp:79-82) —
    /// the side's current per-resource amounts.
    pub side_resources: [i32; 4],
    /// `ps->GetSideRobots()` (CCounter.cpp:88) — robots currently on
    /// the map for this side.
    pub side_robots: i32,
    /// `ps->GetRobotsInStack()` (CCounter.cpp:88) — robots queued in
    /// any production stack for this side.
    pub robots_in_stack: i32,
    /// `ps->GetMaxSideRobots()` (CCounter.cpp:88) — population cap.
    pub max_side_robots: i32,
    /// Active object (if any) is a base — count of items already in
    /// that base's build-stack `m_BS` (CCounter.cpp:92-97). `None` when
    /// the active object isn't a base.
    pub active_base_stack_items: Option<i32>,
}

/// Port of `CIFaceCounter` (CCounter.h:12-44). A small counter widget
/// driving the build-multiplier on the constructor panel.
#[derive(Debug, Clone)]
pub struct CIFaceCounter {
    /// Port of `m_Counter` (CCounter.h:14). Defaults to 1 (DOWN_LIMIT).
    counter: i32,
    /// Port of `m_ButtonUp` enabled state. `true` = `IFACE_NORMAL`,
    /// `false` = `IFACE_DISABLED`. C++ sets these via direct
    /// `m_ButtonUp->SetState(IFACE_DISABLED)` calls.
    pub button_up_enabled: bool,
    /// Port of `m_ButtonDown` enabled state.
    pub button_down_enabled: bool,
}

impl Default for CIFaceCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl CIFaceCounter {
    /// Port of `CIFaceCounter::CIFaceCounter` (CCounter.cpp:13-19).
    pub fn new() -> Self {
        Self {
            counter: DOWN_LIMIT,
            button_up_enabled: true,
            button_down_enabled: true,
        }
    }

    /// Port of `GetCounter` (CCounter.h:22).
    pub fn counter(&self) -> i32 {
        self.counter
    }

    /// Port of `Inc` (CCounter.h:26). Clamps at `UP_LIMIT`.
    pub fn inc(&mut self) -> i32 {
        self.counter += 1;
        if self.counter > UP_LIMIT {
            self.counter = UP_LIMIT;
        }
        self.counter
    }

    /// Port of `Dec` (CCounter.h:27). Clamps at `DOWN_LIMIT`.
    pub fn dec(&mut self) -> i32 {
        self.counter -= 1;
        if self.counter < DOWN_LIMIT {
            self.counter = DOWN_LIMIT;
        }
        self.counter
    }

    /// Port of `ManageButtons` (CCounter.cpp:25-40). Disables `bup` at
    /// `UP_LIMIT`, disables `bdown` at `DOWN_LIMIT`, otherwise enables
    /// both.
    pub fn manage_buttons(&mut self) {
        self.button_up_enabled = self.counter != UP_LIMIT;
        self.button_down_enabled = self.counter != DOWN_LIMIT;
    }

    /// Port of `Reset` (CCounter.h:34). Returns the counter to
    /// `DOWN_LIMIT` and refreshes button enabled state.
    pub fn reset(&mut self) {
        self.counter = DOWN_LIMIT;
        self.manage_buttons();
    }

    /// Port of `CheckUp` (CCounter.cpp:66-99). Disables the up/down
    /// buttons (and zeroes the counter) when the side can't even afford
    /// the current count; disables only `bup` when the side can't
    /// afford `counter+1` units, would exceed the population cap, or
    /// would overflow the active base's build stack (max 6 items).
    pub fn check_up(&mut self, ctx: CheckUpCtx) {
        // CCounter.cpp:79-82
        let mut res_now = [0i32; 4];
        let mut res_next = [0i32; 4];
        for r in Resource::ALL {
            let i = r as usize;
            res_now[i] = ctx.per_unit_price.resources[i] * self.counter;
            res_next[i] = ctx.per_unit_price.resources[i] * (self.counter + 1);
        }

        let enough = |required: &[i32; 4]| -> bool {
            for r in Resource::ALL {
                let i = r as usize;
                if ctx.side_resources[i] < required[i] {
                    return false;
                }
            }
            true
        };

        // CCounter.cpp:83-87 — can't afford even current count.
        if !enough(&res_now) {
            self.button_up_enabled = false;
            self.button_down_enabled = false;
            self.counter = 0;
        }
        // CCounter.cpp:88-90 — can't afford one more, or would exceed
        // population cap.
        if !enough(&res_next)
            || (ctx.side_robots + ctx.robots_in_stack + self.counter >= ctx.max_side_robots)
        {
            self.button_up_enabled = false;
        }

        // CCounter.cpp:92-98 — base stack-overflow check (cap 6).
        if let Some(items) = ctx.active_base_stack_items {
            if (items + self.counter + 1) > 6 {
                self.button_up_enabled = false;
            }
        }
    }

    /// Port of `Disable` (CCounter.h:38). Used when no constructor is
    /// open / no resources are available.
    pub fn disable(&mut self) {
        self.button_up_enabled = false;
        self.button_down_enabled = false;
        self.counter = 0;
    }

    /// Port of `Enable` (CCounter.h:39). Re-enables the buttons and
    /// resets to `DOWN_LIMIT`, then revalidates against the current
    /// resource/stack state.
    pub fn enable(&mut self, ctx: CheckUpCtx) {
        self.button_up_enabled = true;
        self.button_down_enabled = true;
        self.reset();
        self.check_up(ctx);
    }

    /// Port of `Up` callback (CCounter.h:35). The C++ also calls
    /// `MulRes` (CreateSummPrice); the rust caller refreshes the summ
    /// price when it picks up the new counter, so that step is implicit.
    pub fn up(&mut self, ctx: CheckUpCtx) {
        self.inc();
        self.manage_buttons();
        self.check_up(ctx);
    }

    /// Port of `Down` callback (CCounter.h:36).
    pub fn down(&mut self, ctx: CheckUpCtx) {
        self.dec();
        self.manage_buttons();
        self.check_up(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_price(per_unit: i32, resources: i32) -> CheckUpCtx {
        let mut p = UnitPrice::zero();
        for r in Resource::ALL {
            p.resources[r as usize] = per_unit;
        }
        CheckUpCtx {
            per_unit_price: p,
            side_resources: [resources; 4],
            side_robots: 0,
            robots_in_stack: 0,
            max_side_robots: 100,
            active_base_stack_items: None,
        }
    }

    #[test]
    fn defaults_to_down_limit() {
        let c = CIFaceCounter::new();
        assert_eq!(c.counter(), DOWN_LIMIT);
        assert!(c.button_up_enabled);
        assert!(c.button_down_enabled);
    }

    #[test]
    fn inc_clamps_at_up_limit() {
        let mut c = CIFaceCounter::new();
        for _ in 0..10 {
            c.inc();
        }
        assert_eq!(c.counter(), UP_LIMIT);
    }

    #[test]
    fn dec_clamps_at_down_limit() {
        let mut c = CIFaceCounter::new();
        for _ in 0..10 {
            c.dec();
        }
        assert_eq!(c.counter(), DOWN_LIMIT);
    }

    #[test]
    fn manage_buttons_disables_at_limits() {
        let mut c = CIFaceCounter::new();
        c.manage_buttons();
        assert!(!c.button_down_enabled);
        assert!(c.button_up_enabled);
        for _ in 0..UP_LIMIT {
            c.inc();
        }
        c.manage_buttons();
        assert!(!c.button_up_enabled);
        assert!(c.button_down_enabled);
    }

    #[test]
    fn check_up_disables_both_when_cant_afford_now() {
        let mut c = CIFaceCounter::new();
        c.check_up(ctx_with_price(100, 50));
        assert!(!c.button_up_enabled);
        assert!(!c.button_down_enabled);
        assert_eq!(c.counter(), 0);
    }

    #[test]
    fn check_up_disables_only_up_when_cant_afford_next() {
        let mut c = CIFaceCounter::new();
        c.check_up(ctx_with_price(100, 100));
        assert!(!c.button_up_enabled);
        assert!(c.button_down_enabled);
    }

    #[test]
    fn check_up_disables_up_at_population_cap() {
        let mut c = CIFaceCounter::new();
        let mut ctx = ctx_with_price(0, 1000);
        ctx.side_robots = 99;
        c.check_up(ctx);
        assert!(!c.button_up_enabled);
    }

    #[test]
    fn check_up_disables_up_at_base_stack_cap() {
        let mut c = CIFaceCounter::new();
        let mut ctx = ctx_with_price(0, 1000);
        ctx.active_base_stack_items = Some(5);
        c.check_up(ctx);
        assert!(!c.button_up_enabled);
    }
}
