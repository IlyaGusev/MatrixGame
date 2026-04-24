use matrixgame_rs::matrix_game::interface::interface::DESIGN_H;
use matrixgame_rs::matrix_game::interface::{Click, IFaceList};
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let dat = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&dat).unwrap();
    let mut list = IFaceList::load_default_panels(&stor);
    // Show Base panel (constructor active).
    if let Some(p) = list.panels.iter_mut().find(|p| p.name == "Base") {
        p.visible = true;
    }
    let w = 1024.0;
    let h = 768.0;
    let scale = (h / DESIGN_H).max(0.1);
    println!("DESIGN_H={DESIGN_H}, scale={scale}");
    // Find cocan rect
    for p in &list.panels {
        if p.name != "Base" {
            continue;
        }
        let [px, py] = p.resolved_pos(w, h, scale);
        println!("Base panel resolved at ({px:.0},{py:.0})");
        for e in &p.elements {
            if matches!(e.name.as_str(), "cocan" | "pich" | "pi1") {
                let [x, y, ew, eh] = e.rect_in_panel([px, py], scale);
                let cx = x + ew / 2.0;
                let cy = y + eh / 2.0;
                println!(
                    "  {} rect=({x:.0},{y:.0},{ew:.0},{eh:.0}) center=({cx:.0},{cy:.0}) visible={}",
                    e.name,
                    e.visible()
                );
            }
        }
    }
    // Try clicking cocan
    let cocan_rect = list.panels.iter().find(|p| p.name == "Base").and_then(|p| {
        let [px, py] = p.resolved_pos(w, h, scale);
        p.elements.iter().find(|e| e.name == "cocan").map(|e| {
            let [x, y, ew, eh] = e.rect_in_panel([px, py], scale);
            (x + ew / 2.0, y + eh / 2.0)
        })
    });
    if let Some((cx, cy)) = cocan_rect {
        println!("\nSimulating click at cocan center ({cx:.0},{cy:.0})");
        // First mark cocan visible (constructor open) and call refresh visibility
        // Then click.
        for p in list.panels.iter_mut() {
            if p.name == "Base" {
                for e in p.elements.iter_mut() {
                    e.set_visible(true);
                }
            }
        }
        let down = list.on_mouse_down(cx, cy, w, h);
        println!(
            "  on_mouse_down → consumed={down}, pressed={:?}",
            list.pressed
        );
        let up = list.on_mouse_up(cx, cy, w, h);
        println!("  on_mouse_up → click={:?}", up);
    }
}
