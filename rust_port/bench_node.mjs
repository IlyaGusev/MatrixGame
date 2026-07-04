import { readFileSync } from "fs";
import { bench_battle } from "./pkg-bench/matrixgame_rs.js";
const pkg = readFileSync("../Data/robots.pkg");
const dat = readFileSync("../Data/robots.dat");
console.log(bench_battle(pkg, dat));
