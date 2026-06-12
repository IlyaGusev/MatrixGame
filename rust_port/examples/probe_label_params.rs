// Dump full per-label config: Params / Font / Color / State for the
// Base panel — to understand what real-data the original drives the
// SetStateLabelParams call with.
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let bytes = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&bytes).unwrap();
    let if_base = stor.block_record("if", "Base").unwrap();
    // pbp2 in the C++ is the per-element block — under "if/Base" each
    // element record's children include `Labels` (named-empty) blocks.
    // Walk children "2"/"3" of if/Base.
    let names = stor.get_buf(&if_base, "2").unwrap();
    let recs = stor.get_buf(&if_base, "3").unwrap();
    println!("if/Base has {} child blocks", names.arrays_count());
    let mut interesting = 0;
    for i in 0..names.arrays_count() {
        let kind = names.get_as_wstr(i);
        let rec = recs.get_as_wstr(i);
        if kind != "Button" && kind != "Static" {
            continue;
        }
        let elem_name = stor.block_param(&rec, "Name").unwrap_or_default();
        if ![
            "cobuild",
            "cocan",
            "chasl",
            "weapl",
            "buildl",
            "it_label1",
            "it_label2",
        ]
        .contains(&elem_name.as_str())
        {
            continue;
        }
        interesting += 1;
        // Print the element's child blocks
        let Some(child_names) = stor.get_buf(&rec, "2") else {
            continue;
        };
        let Some(child_recs) = stor.get_buf(&rec, "3") else {
            continue;
        };
        println!(
            "\n--- {kind} '{elem_name}' (children: {}) ---",
            child_names.arrays_count()
        );
        for j in 0..child_names.arrays_count() {
            let cn = child_names.get_as_wstr(j);
            let cr = child_recs.get_as_wstr(j);
            // Look for Labels or per-state blocks
            if cn != "Labels" {
                continue;
            }
            // Labels is a CBlockPar of named blocks (one per label entry).
            let Some(lbl_names) = stor.get_buf(&cr, "2") else {
                continue;
            };
            let Some(lbl_recs) = stor.get_buf(&cr, "3") else {
                continue;
            };
            for k in 0..lbl_names.arrays_count() {
                let lbl_name = lbl_names.get_as_wstr(k);
                let lbl_rec = lbl_recs.get_as_wstr(k);
                let params = stor.block_param(&lbl_rec, "Params").unwrap_or_default();
                let state = stor.block_param(&lbl_rec, "State").unwrap_or_default();
                let font = stor.block_param(&lbl_rec, "Font").unwrap_or_default();
                let color = stor.block_param(&lbl_rec, "Color").unwrap_or_default();
                println!(
                    "  Labels[{k}].name='{lbl_name}' Params=[{params}] State={state} Font={font} Color={color}"
                );
            }
        }
    }
    println!("\n{interesting} interesting elements");
}
