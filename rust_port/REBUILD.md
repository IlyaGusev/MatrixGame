# Rebuild & Rerun

All commands run from `rust_port/`.

## Quick rebuild (dev, ~2 s incremental)

```bash
wasm-pack build --dev --target web --out-dir pkg
```

Then bump the cache-bust version in `index.html`:

```bash
sed -i -E 's|(matrixgame_rs\.js\?v=)([0-9]+)|echo "\1$((\2+1))"|e' index.html
```

…or edit the `?v=N` by hand.

Release build (slower, ~46 s, smaller/faster wasm):

```bash
wasm-pack build --target web --out-dir pkg
```

## Asset bundle (only when map / object / texture data changes)

The WASM build reads `assets/atoll.bundle` at startup. Regenerate it after
editing `pack_bundle.rs` or when adding new VO meshes / textures:

```bash
cargo run --release --example pack_bundle
```

## Serve locally

```bash
fuser -k 8081/tcp 2>/dev/null; nohup python3 -m http.server 8081 > /dev/null 2>&1 &
```

Open http://localhost:8081.

## Full flow (rebuild + rerun)

```bash
wasm-pack build --dev --target web --out-dir pkg && \
  sed -i -E 's|(matrixgame_rs\.js\?v=)([0-9]+)|echo "\1$((\2+1))"|e' index.html && \
  fuser -k 8081/tcp 2>/dev/null; \
  nohup python3 -m http.server 8081 > /dev/null 2>&1 &
```

Reload http://localhost:8081 in the browser.

## Native run (debug)

Native build requires a display; it bypasses the bundle and reads directly
from `../Data/robots.pkg`:

```bash
cargo run
# or: RUST_LOG=info cargo run
```
