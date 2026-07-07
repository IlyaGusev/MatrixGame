//! Dump the `Cursors` block and exercise the MatrixCursor port
//! (parse → select → takt → sprite) against the real robots.dat.
use matrixgame_rs::matrix_game::cursor::{self, MatrixCursor};
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    let Some(rec) = stor.block_record("da", "Cursors") else {
        println!("no Cursors block");
        return;
    };
    for key in [
        cursor::CURSOR_ARROW,
        cursor::CURSOR_CROSS_BLUE,
        cursor::CURSOR_CROSS_RED,
        cursor::CURSOR_CROSS_YELLOW,
        cursor::CURSOR_STAR,
    ] {
        println!("  {} = {:?}", key, stor.block_param(&rec, key).as_deref());
    }

    let mut c = MatrixCursor::new(&stor);
    c.pos = [100.0, 100.0];
    for name in [
        cursor::CURSOR_ARROW,
        cursor::CURSOR_CROSS_BLUE,
        cursor::CURSOR_CROSS_RED,
        cursor::CURSOR_CROSS_YELLOW,
        cursor::CURSOR_STAR,
    ] {
        c.select(name);
        println!("== {} -> {:?}", name, c.tex_path());
        // Walk 2s of animation, print a few sampled frames.
        for step in 0..40 {
            c.takt(50);
            if step % 10 == 9 {
                let s = c.sprite(128, 128, 1.0).unwrap();
                println!(
                    "  t={:>4}ms quad=({:>5.1},{:>5.1} {}x{}) uv=[{:.4},{:.4},{:.4},{:.4}]",
                    (step + 1) * 50,
                    s.x,
                    s.y,
                    s.w,
                    s.h,
                    s.uv[0],
                    s.uv[1],
                    s.uv[2],
                    s.uv[3]
                );
            }
        }
    }
}
