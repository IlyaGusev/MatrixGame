use matrixgame_rs::matrix_game::interface::CInterface;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let bytes = std::fs::read("../Data/robots.dat").expect("robots.dat next to ../Data");
    let stor = Storage::from_bytes(&bytes).unwrap();

    // Load each panel like form_game does and dump per-element labels.
    for panel in ["Base", "Main", "Top"] {
        let Some(p) = CInterface::load(&stor, panel) else {
            println!("panel {panel}: not loaded");
            continue;
        };
        println!("\n=== panel {} ({} elements) ===", p.name, p.elements.len());
        let mut labelled = 0;
        for e in &p.elements {
            if e.labels.is_empty() {
                continue;
            }
            labelled += 1;
            for l in &e.labels {
                println!(
                    "  {} [{:?}] '{}'  font={} color=({},{},{},{}) align=({},{}) pos=({},{})+sme=({},{})",
                    e.name,
                    l.state,
                    l.text,
                    l.font,
                    l.color[0],
                    l.color[1],
                    l.color[2],
                    l.color[3],
                    l.align_x,
                    l.align_y,
                    l.x,
                    l.y,
                    l.sme_x,
                    l.sme_y,
                );
            }
        }
        println!("-- {labelled} elements have labels --");
    }
}
