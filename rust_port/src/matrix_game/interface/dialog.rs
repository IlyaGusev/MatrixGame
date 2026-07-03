//! Dialog-mode state — port of `CMatrixMap::EnterDialogMode` /
//! `LeaveDialogMode` (MatrixMap.cpp:3261-3574) and the hint-button
//! handlers around them. The form-game layer owns the build/teardown
//! glue (it needs the renderer + panels); this module holds the pure
//! state + the stat-dialog replacement formatting so both are unit-
//! testable headless.

use std::collections::HashMap;

use crate::matrix_lib::base::storage::Storage;

use super::hint::Hint;

/// Element names of the `if/Hints` panel buttons
/// (StringConstants.hpp:1019-1026).
pub const HINT_OK: &str = "hint_ok";
pub const HINT_CANCEL: &str = "hint_cancel";
pub const HINT_CANCEL_MENU: &str = "hint_cmenu";
pub const HINT_CONTINUE: &str = "hint_cont";
pub const HINT_SURRENDER: &str = "hint_surr";
pub const HINT_EXIT: &str = "hint_exit";
pub const HINT_RESET: &str = "hint_reset";
pub const HINT_HELP: &str = "hint_help";

/// What a hint button does when pressed — replaces the C++ function-
/// pointer handlers (`DialogButtonHandler`, MatrixMap.cpp:3285-3431).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogAction {
    /// `OkHandler` — close the dialog and resume. Also used for
    /// HINT_CONTINUE / HINT_CANCEL_MENU / the StatD OK / HINT_HELP
    /// (HelpRequestHandler launches Manual.exe after leaving the
    /// dialog — no manual to launch in the port, so only the leave
    /// remains).
    LeaveDialog,
    /// `ExitRequestHandler` — open the Exit confirmation.
    ExitRequest,
    /// `ResetRequestHandler`.
    ResetRequest,
    /// `SurrenderRequestHandler`.
    SurrenderRequest,
    /// `OkExitHandler` — g_ExitState=1, then the Stat dialog.
    ConfirmExit,
    /// `OkResetHandler` — g_MatrixMap->Restart().
    ConfirmReset,
    /// `OkSurrenderHandler` — g_ExitState=4 + negate enemy STAT_TIME,
    /// then the Stat dialog.
    ConfirmSurrender,
    /// `ConfirmCancelHandler` — drop the confirmation sub-dialog.
    ConfirmCancel,
    /// `OkExitWinHandler` — g_ExitState=3, then the Stat dialog.
    OkWin,
    /// `OkExitLooseHandler` — g_ExitState=2, then the Stat dialog.
    OkLoose,
    /// `OkJustExitHandler` — Stat OK after win/lose/exit: leave the
    /// game loop entirely.
    OkStatExit,
    /// `OkHandler` on the StatD (mid-game stats) variant: back to the
    /// game.
    OkStatBack,
}

/// One created hint button — mirrors the `CreateHintButton` calls:
/// the Hints-panel element `elem` shown at design-space `pos` with
/// `action` as its press handler. `disabled` ports the
/// Disable/EnableMainMenuButton state while a confirmation is up.
#[derive(Debug, Clone)]
pub struct DialogButton {
    pub elem: &'static str,
    pub action: DialogAction,
    /// Design-space (1024×768) position for the element's top-left.
    pub pos: (f32, f32),
    pub disabled: bool,
}

/// Port of the `m_DialogModeName` + `m_DialogModeHints` pair plus the
/// button bookkeeping `g_IFaceList` keeps on the Hints panel.
#[derive(Debug, Clone, Default)]
pub struct DialogState {
    /// `m_DialogModeName` — "Menu" / "Win" / "Loose" / "Stat" / "StatD".
    pub name: String,
    /// `m_DialogModeHints` — built widgets in creation order. The
    /// confirmation sub-dialog appends to this (MatrixMap.cpp:3390).
    pub hints: Vec<Hint>,
    pub buttons: Vec<DialogButton>,
    /// Buttons added by the confirmation (so cancel can remove exactly
    /// them).
    confirm_buttons: usize,
}

impl DialogState {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// `m_DialogModeHints.Len() > 4` — more than one hint means a
    /// confirmation sub-dialog is up (only meaningful for "Menu",
    /// which has exactly one base hint).
    pub fn has_confirmation(&self) -> bool {
        self.confirm_buttons > 0
    }

    pub fn is_menu(&self) -> bool {
        self.name == "Menu"
    }

    /// `m_DialogModeHints.Len() > 4 || name != Menu` — the ENTER-key
    /// gate at MatrixFormGame.cpp:1172-1177.
    pub fn enter_presses_ok(&self) -> bool {
        self.hints.len() > 1 || !self.is_menu()
    }

    pub fn action_of(&self, elem: &str) -> Option<DialogAction> {
        self.buttons
            .iter()
            .find(|b| b.elem == elem && !b.disabled)
            .map(|b| b.action)
    }

    /// Port of `CreateConfirmation` (MatrixMap.cpp:3359-3392):
    /// disable the menu buttons, append the confirmation hint, create
    /// HINT_OK (`ok_action`) + HINT_CANCEL at its copy positions.
    pub fn open_confirmation(
        &mut self,
        hint: Hint,
        copy_pos_design: &[(f32, f32)],
        ok_action: DialogAction,
    ) {
        for b in self.buttons.iter_mut() {
            b.disabled = true;
        }
        self.confirm_buttons = 0;
        if let Some(&p) = copy_pos_design.first() {
            self.buttons.push(DialogButton {
                elem: HINT_OK,
                action: ok_action,
                pos: p,
                disabled: false,
            });
            self.confirm_buttons += 1;
        }
        if let Some(&p) = copy_pos_design.get(1) {
            self.buttons.push(DialogButton {
                elem: HINT_CANCEL,
                action: DialogAction::ConfirmCancel,
                pos: p,
                disabled: false,
            });
            self.confirm_buttons += 1;
        }
        self.hints.push(hint);
    }

    /// Port of `ConfirmCancelHandler` (MatrixMap.cpp:3339-3355): drop
    /// the confirmation hint + its OK/Cancel buttons, re-enable the
    /// menu buttons.
    pub fn cancel_confirmation(&mut self) {
        if self.confirm_buttons == 0 {
            return;
        }
        self.hints.pop();
        let keep = self.buttons.len() - self.confirm_buttons;
        self.buttons.truncate(keep);
        self.confirm_buttons = 0;
        for b in self.buttons.iter_mut() {
            b.disabled = false;
        }
    }
}

/// Per-side stat snapshot feeding the Stat dialog replacements.
#[derive(Debug, Clone)]
pub struct SideStatRow {
    /// `_p_` for the player, `_<first letter of side name>_` otherwise
    /// (MatrixLogic.cpp:2537-2544).
    pub prefix: String,
    /// `m_Statistic` in `EStat` order: robot_build, robot_kill,
    /// turret_build, turret_kill, building_kill, time(ms).
    pub stats: [i32; 6],
}

/// Side-id → `_x_` prefix from the `Side` block of MatrixData: first
/// letter of the side name, lowercased, wrapped in underscores
/// (MatrixLogic.cpp:2540-2543). The player side ignores this and uses
/// `_p_`.
pub fn load_side_prefixes(storage: &Storage) -> HashMap<i32, String> {
    let mut out = HashMap::new();
    let Some(rec) = storage.block_record("da", "Side") else {
        return out;
    };
    let (Some(keys), Some(vals)) = (storage.get_buf(&rec, "0"), storage.get_buf(&rec, "1")) else {
        return out;
    };
    for i in 0..keys.arrays_count().min(vals.arrays_count()) {
        let Ok(id) = keys.get_as_wstr(i).parse::<i32>() else {
            continue;
        };
        let val = vals.get_as_wstr(i);
        if let Some(c) = val.chars().next() {
            out.insert(id, format!("_{}_", c.to_lowercase()));
        }
    }
    out
}

/// `hh:mm:ss` with 2-digit zero padding (MatrixLogic.cpp:2558-2567).
fn format_time(total_secs: i32) -> String {
    let hours = total_secs / 3600;
    let mins = (total_secs - hours * 3600) / 60;
    let secs = total_secs - hours * 3600 - mins * 60;
    format!("{hours:02}:{mins:02}:{secs:02}")
}

/// Build the `Replace` pairs the Stat template consumes — port of the
/// MMFLAG_STAT_DIALOG block at MatrixLogic.cpp:2531-2586. A negative
/// STAT_TIME marks the winner: shown positive and wrapped in green.
pub fn build_stat_replacements(rows: &[SideStatRow]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for row in rows {
        let time_ms = row.stats[5];
        if time_ms == 0 {
            out.push((format!("{}c6", row.prefix), String::new()));
            continue;
        }
        let mut time = time_ms / 1000;
        let winner = time < 0;
        if winner {
            time = -time;
        }
        let hms = format_time(time);
        let c6 = if winner {
            format!("<color=0,255,0>{hms}</color>")
        } else {
            hms
        };
        out.push((format!("{}c6", row.prefix), c6));
        for (i, key) in ["c1", "c2", "c3", "c4", "c5"].iter().enumerate() {
            out.push((format!("{}{}", row.prefix, key), row.stats[i].to_string()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(rows: &[SideStatRow]) -> HashMap<String, String> {
        build_stat_replacements(rows).into_iter().collect()
    }

    #[test]
    fn stat_time_formats_hh_mm_ss() {
        let m = pairs(&[SideStatRow {
            prefix: "_p_".into(),
            stats: [3, 1, 4, 1, 5, 3_725_000], // 1h 2m 5s
        }]);
        assert_eq!(m["_p_c6"], "01:02:05");
        assert_eq!(m["_p_c1"], "3");
        assert_eq!(m["_p_c2"], "1");
        assert_eq!(m["_p_c3"], "4");
        assert_eq!(m["_p_c4"], "1");
        assert_eq!(m["_p_c5"], "5");
    }

    #[test]
    fn stat_negative_time_marks_winner_green() {
        let m = pairs(&[SideStatRow {
            prefix: "_r_".into(),
            stats: [0, 0, 0, 0, 0, -61_000],
        }]);
        assert_eq!(m["_r_c6"], "<color=0,255,0>00:01:01</color>");
    }

    #[test]
    fn stat_zero_time_clears_row() {
        let m = pairs(&[SideStatRow {
            prefix: "_g_".into(),
            stats: [9, 9, 9, 9, 9, 0],
        }]);
        assert_eq!(m["_g_c6"], "");
        assert!(!m.contains_key("_g_c1"), "c1..c5 stay unset when time==0");
    }

    fn menu_dialog() -> DialogState {
        // The Menu dialog's 6 buttons in copy-pos order
        // (MatrixMap.cpp:3505-3565).
        let buttons = [
            (HINT_CONTINUE, DialogAction::LeaveDialog),
            (HINT_RESET, DialogAction::ResetRequest),
            (HINT_HELP, DialogAction::LeaveDialog),
            (HINT_SURRENDER, DialogAction::SurrenderRequest),
            (HINT_EXIT, DialogAction::ExitRequest),
            (HINT_CANCEL_MENU, DialogAction::LeaveDialog),
        ]
        .into_iter()
        .map(|(elem, action)| DialogButton {
            elem,
            action,
            pos: (0.0, 0.0),
            disabled: false,
        })
        .collect();
        DialogState {
            name: "Menu".into(),
            hints: vec![Hint {
                parts: Vec::new(),
                total_w: 100,
                total_h: 100,
                border_id: -1,
                otstup: [0; 4],
                screen_x: 0.0,
                screen_y: 0.0,
                sound_out: None,
            }],
            buttons,
            confirm_buttons: 0,
        }
    }

    /// Enter menu → confirm exit → cancel: the menu buttons disable
    /// while the confirmation is up and recover on cancel; the
    /// OK/Cancel pair appears and disappears with it.
    #[test]
    fn menu_confirmation_round_trip() {
        let mut d = menu_dialog();
        assert!(!d.has_confirmation());
        assert!(!d.enter_presses_ok(), "bare menu must not react to ENTER");
        assert_eq!(d.action_of(HINT_EXIT), Some(DialogAction::ExitRequest));

        let confirm_hint = d.hints[0].clone();
        d.open_confirmation(
            confirm_hint,
            &[(10.0, 10.0), (60.0, 10.0)],
            DialogAction::ConfirmExit,
        );
        assert!(d.has_confirmation());
        assert!(d.enter_presses_ok());
        assert_eq!(d.hints.len(), 2);
        assert_eq!(d.action_of(HINT_OK), Some(DialogAction::ConfirmExit));
        assert_eq!(d.action_of(HINT_CANCEL), Some(DialogAction::ConfirmCancel));
        // Menu buttons are disabled — action_of skips them.
        assert_eq!(d.action_of(HINT_EXIT), None);

        d.cancel_confirmation();
        assert!(!d.has_confirmation());
        assert_eq!(d.hints.len(), 1);
        assert_eq!(d.action_of(HINT_OK), None);
        assert_eq!(d.action_of(HINT_CANCEL), None);
        assert_eq!(d.action_of(HINT_EXIT), Some(DialogAction::ExitRequest));
    }
}
