//! Inspect robots.dat's hint-related blocks: template names, replace
//! keys, and which UI elements carry a `Hint` param.
//!
//! Run from the repo root:
//! ```
//! cargo run --example probe_hints
//! ```
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let bytes = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let stor = Storage::from_bytes(&bytes).expect("storage parse");

    // AllLabels/Turrets — per-turret name+range labels used by the
    // BuildTurret hint (needed for `_turret_name` / `_turret_range`).
    if let Some(all) = stor.block_record("da", "AllLabels") {
        if let Some(turr) = stor.block_record(&all, "Turrets") {
            if let (Some(k), Some(v)) = (stor.get_buf(&turr, "0"), stor.get_buf(&turr, "1")) {
                let n = k.arrays_count();
                println!("AllLabels/Turrets: {} entries", n);
                for i in 0..n.min(12) {
                    println!("  {} = {}", k.get_as_wstr(i), v.get_as_wstr(i));
                }
            }
        }
    }

    // Probe specific `_hint` keys in AllLabels/Replaces — the source
    // text that the template's `[key]` tokens resolve to.
    if let Some(all) = stor.block_record("da", "AllLabels") {
        if let Some(rr) = stor.block_record(&all, "Replaces") {
            if let (Some(keys), Some(vals)) = (stor.get_buf(&rr, "0"), stor.get_buf(&rr, "1")) {
                for target in ["titan_hint", "electronics_hint", "energy_hint", "plasma_hint", "robots_hint", "bt_name_hint", "bt_params_hint", "bt_res1_hint", "bt_res2_hint", "bt_res3_hint", "bt_res4_hint"] {
                    for i in 0..keys.arrays_count() {
                        if keys.get_as_wstr(i) == target {
                            let v = vals.get_as_wstr(i);
                            let short: String = v.chars().take(220).collect();
                            println!("AllLabels/Replaces '{target}' = {short}");
                        }
                    }
                }
            }
        }
    }

    // Interesting template names used by the resource-bar hints (Top
    // panel) — check they exist so we know the builder finds them.
    if let Some(t_rec) = stor.block_record("da", "Templates") {
        if let Some(keys) = stor.get_buf(&t_rec, "0") {
            let vals = stor.get_buf(&t_rec, "1");
            for target in ["Titan", "Electronics", "Energy", "Plasma", "Robots", "HistoryNext"] {
                for i in 0..keys.arrays_count() {
                    if keys.get_as_wstr(i) == target {
                        let v = vals
                            .map(|vb| vb.get_as_wstr(i))
                            .unwrap_or_default();
                        println!("Template '{target}' = {v}");
                    }
                }
            }
        }
    }

    for (label, path) in [
        ("Templates", vec!["da", "Templates"]),
        ("Replaces (top)", vec!["da", "Replaces"]),
        ("AllLabels/Replaces", vec!["da", "AllLabels", "Replaces"]),
    ] {
        let mut rec_opt: Option<String> = None;
        let mut cur = "".to_string();
        for (i, name) in path.iter().enumerate() {
            let parent = if i == 0 { *name } else { cur.as_str() };
            if i == 0 {
                cur = parent.to_string();
                continue;
            }
            let Some(next) = stor.block_record(parent, name) else {
                println!("{label}: missing at step '{name}' (parent='{parent}')");
                rec_opt = None;
                break;
            };
            cur = next.clone();
            rec_opt = Some(next);
        }
        let Some(rec) = rec_opt else { continue };
        let keys = stor.get_buf(&rec, "0");
        let vals = stor.get_buf(&rec, "1");
        let n = keys.map(|k| k.arrays_count()).unwrap_or(0);
        println!("{label}: rec='{rec}' entries={n}");
        if let (Some(k), Some(v)) = (keys, vals) {
            for i in 0..n.min(10) {
                let ks = k.get_as_wstr(i);
                let vs = v.get_as_wstr(i);
                let preview: String = vs.chars().take(90).collect();
                println!("  [{i}] {ks} = {preview}{}", if vs.len() > 90 { " …" } else { "" });
            }
        }
    }

    // Walk every `if/<Panel>` and list elements that set Hint=.
    // `if` is a TOP-LEVEL record in robots.dat (not nested under `da`) —
    // CInterface::load uses `block_record("if", name)` directly.
    {
        let names = stor.get_buf("if", "2");
        let recs = stor.get_buf("if", "3");
        let Some(names) = names else {
            println!("if: no top-level 'if/2' column");
            return;
        };
        let Some(recs) = recs else { return };
        let n = names.arrays_count().min(recs.arrays_count());
        for i in 0..n {
            let panel_name = names.get_as_wstr(i);
            let panel_rec = recs.get_as_wstr(i);
            let Some(enames) = stor.get_buf(&panel_rec, "2") else { continue };
            let Some(erecs) = stor.get_buf(&panel_rec, "3") else { continue };
            let en = enames.arrays_count().min(erecs.arrays_count());
            let mut found: Vec<(String, String)> = Vec::new();
            for ei in 0..en {
                let child_rec = erecs.get_as_wstr(ei);
                let elem_name = stor.block_param(&child_rec, "Name").unwrap_or_default();
                if let Some(hint) = stor.block_param(&child_rec, "Hint") {
                    if !hint.is_empty() {
                        found.push((elem_name, hint));
                    }
                }
            }
            if !found.is_empty() {
                println!("if/{panel_name}: {} hinted elements", found.len());
                for (n, h) in found.iter().take(20) {
                    println!("  {n} Hint={h}");
                }
            }
        }
    }
}
