# Porting MatrixGame to Rust + wgpu → Browser (WASM)

A step-by-step guide to rewriting the Space Rangers 2 Planetary Battles engine
in Rust and running it in the browser via WebAssembly.

---

## Overview

**Goal:** Replace the C++ / DirectX 9 engine with a Rust codebase that compiles
to both native (desktop) and WASM (browser) targets using the same source.

**Key insight:** `wgpu` abstracts over DirectX 12 / Vulkan / Metal on desktop
and over WebGL 2 / WebGPU in the browser. You write the rendering code once.

```
MatrixGame (C++, DirectX 9, Windows-only)
        ↓  rewrite
Rust + wgpu + winit
        ↓  cargo build
   native binary          WASM + WebGL 2 / WebGPU
  (Windows/Linux/macOS)        (any browser)
```

---

## Prerequisites

### Tools to install

```bash
# 1. Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# 2. WASM target
rustup target add wasm32-unknown-unknown

# 3. wasm-pack — builds and packages WASM for the web
cargo install wasm-pack

# 4. A simple static file server for local testing
cargo install miniserve          # or: npm install -g serve

# 5. (Optional) cargo-watch for live reloading during development
cargo install cargo-watch
```

### Verify

```bash
rustc --version        # rustc 1.77+
wasm-pack --version    # 0.12+
```

---

## Project Structure

```
matrixgame-rs/
├── Cargo.toml
├── index.html               ← browser entry point
├── src/
│   ├── main.rs              ← native entry point (winit event loop)
│   ├── lib.rs               ← WASM entry point (exported to JS)
│   ├── app.rs               ← shared application state + update loop
│   ├── renderer/
│   │   ├── mod.rs
│   │   ├── context.rs       ← wgpu Device, Queue, Surface
│   │   ├── pipeline.rs      ← render pipelines (shaders, vertex layout)
│   │   ├── terrain.rs       ← terrain mesh + texture
│   │   ├── units.rs         ← unit sprites / 3D models
│   │   └── particles.rs     ← explosion / smoke particles
│   ├── game/
│   │   ├── mod.rs
│   │   ├── world.rs         ← game state (units, buildings, resources)
│   │   ├── ai.rs            ← bot logic (ported from MatrixGame AI)
│   │   └── script.rs        ← scripting bridge (rhai)
│   ├── assets/
│   │   ├── mod.rs
│   │   ├── loader.rs        ← async asset loading (different on WASM vs native)
│   │   └── pkg_reader.rs    ← parser for SR2 .pkg resource archives
│   └── platform/
│       ├── mod.rs
│       ├── native.rs        ← std::fs, std::time
│       └── web.rs           ← web-sys, js-sys, fetch API
├── shaders/
│   ├── terrain.wgsl
│   ├── unit.wgsl
│   └── particle.wgsl
└── assets/                  ← unpacked SR2 resources (gitignored)
```

---

## Cargo.toml

```toml
[package]
name    = "matrixgame-rs"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]   # cdylib needed for WASM

[dependencies]
wgpu        = "22"
winit       = { version = "0.30", features = ["rwh_06"] }
bytemuck    = { version = "1", features = ["derive"] }
glam        = "0.29"              # linear algebra (Vec3, Mat4, …)
image       = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
log         = "0.4"
anyhow      = "1"
rhai        = "1"                 # scripting engine for SR2 scripts

# Async runtime — tiny, works on both native and WASM
[dependencies.pollster]
version  = "0.3"
features = ["macro"]

# ── Native-only ──────────────────────────────────────────────────────────────
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = "0.11"
tokio      = { version = "1", features = ["rt", "fs"] }

# ── WASM-only ────────────────────────────────────────────────────────────────
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen       = "0.2"
wasm-bindgen-futures = "0.4"
web-sys            = { version = "0.3", features = [
    "Window", "Document", "HtmlCanvasElement",
    "WebGl2RenderingContext", "Performance",
    "Request", "RequestInit", "Response",
    "Blob", "File", "FileReader",
] }
js-sys             = "0.3"
console_log        = "0.2"
console_error_panic_hook = "0.1"

[profile.release]
opt-level = "z"     # minimize WASM binary size
lto       = true
```

---

## Step 1: wgpu Context

The context is almost identical for native and WASM — wgpu hides the difference.

```rust
// src/renderer/context.rs

use wgpu::*;
use winit::window::Window;

pub struct GfxContext {
    pub surface:  Surface<'static>,
    pub device:   Device,
    pub queue:    Queue,
    pub config:   SurfaceConfiguration,
}

impl GfxContext {
    pub async fn new(window: &Window) -> Self {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),   // picks WebGL2 in browser automatically
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference:       PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let size   = window.inner_size();
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = SurfaceConfiguration {
            usage:        TextureUsages::RENDER_ATTACHMENT,
            format,
            width:        size.width,
            height:       size.height,
            present_mode: PresentMode::AutoVsync,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self { surface, device, queue, config }
    }
}
```

---

## Step 2: Platform Abstraction for Time and I/O

`std::time::Instant` does not exist on WASM. Use the `instant` crate or abstract
it yourself:

```rust
// src/platform/mod.rs

#[cfg(not(target_arch = "wasm32"))]
pub fn now_secs() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
pub fn now_secs() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
        / 1000.0
}
```

---

## Step 3: Asset Loading

SR2 keeps everything in `.pkg` archives. You need to either pre-unpack them
or write a reader. Loading is async and works differently on each platform.

```rust
// src/assets/loader.rs

pub async fn load_bytes(path: &str) -> anyhow::Result<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(tokio::fs::read(path).await?)
    }

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, Response};

        let opts = RequestInit::new();
        let request = Request::new_with_str_and_init(path, &opts)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let window   = web_sys::window().unwrap();
        let response = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let response: Response = response.dyn_into().unwrap();
        let buffer = JsFuture::from(response.array_buffer().unwrap())
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        Ok(js_sys::Uint8Array::new(&buffer).to_vec())
    }
}
```

---

## Step 4: WASM Entry Point

```rust
// src/lib.rs  — compiled only when targeting wasm32

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub async fn run() {
    // Better panic messages in the browser console
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    // Grab <canvas id="game-canvas"> from the page
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("game-canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();

    // Hand off to the shared application loop
    crate::app::run_on_canvas(canvas).await;
}
```

---

## Step 5: Shared Application Loop

```rust
// src/app.rs

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

pub async fn run_on_canvas(
    #[cfg(target_arch = "wasm32")]
    canvas: web_sys::HtmlCanvasElement,
) {
    let event_loop = EventLoop::new().unwrap();

    #[allow(unused_mut)]
    let mut builder = WindowBuilder::new().with_title("MatrixGame");

    // Attach the winit window to the existing <canvas> element
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowBuilderExtWebSys;
        builder = builder.with_canvas(Some(canvas));
    }

    let window = builder.build(&event_loop).unwrap();
    let ctx    = crate::renderer::context::GfxContext::new(&window).await;
    let mut game = crate::game::world::World::new();

    let mut last_time = crate::platform::now_secs();

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                // resize surface …
                let _ = size;
            }
            Event::AboutToWait => {
                let now   = crate::platform::now_secs();
                let dt    = (now - last_time) as f32;
                last_time = now;

                game.update(dt);
                // render …
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
```

---

## Step 6: WGSL Shaders

Replace the old DirectX 9 fixed-function pipeline rendering with WGSL shaders
(WebGPU Shading Language). WGSL works on both desktop wgpu and the browser.
The original codebase has no standalone shader files — rendering is done via
D3D9 fixed-function calls and inline state setup.

```wgsl
// shaders/terrain.wgsl

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) uv       : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0)       uv            : vec2<f32>,
};

struct Uniforms {
    view_proj : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(0) var t_diffuse : texture_2d<f32>;
@group(1) @binding(1) var s_diffuse : sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
```

---

## Step 7: Vertex Layout (matching the C++ structs)

```rust
// src/renderer/pipeline.rs

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TerrainVertex {
    pub position: [f32; 3],
    pub uv:       [f32; 2],
}

impl TerrainVertex {
    // Describes the vertex buffer layout to wgpu
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &[
                // position @ location 0
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // uv @ location 1
                wgpu::VertexAttribute {
                    offset:          mem::size_of::<[f32; 3]>() as _,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}
```

---

## Step 8: HTML Page

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MatrixGame</title>
  <style>
    body   { margin: 0; background: #000; display: flex;
             justify-content: center; align-items: center; height: 100vh; }
    canvas { display: block; }
  </style>
</head>
<body>
  <canvas id="game-canvas" width="1024" height="768"></canvas>

  <script type="module">
    import init from "./pkg/matrixgame_rs.js";
    await init();           // calls the #[wasm_bindgen(start)] fn automatically
  </script>
</body>
</html>
```

---

## Step 9: Building

### Native (for fast iteration)

```bash
cargo run
```

### WASM (for the browser)

```bash
wasm-pack build --target web --out-dir pkg

# Serve locally
miniserve . --index index.html
# open http://localhost:8080
```

`wasm-pack` produces:
```
pkg/
├── matrixgame_rs_bg.wasm   ← the compiled binary
├── matrixgame_rs.js        ← JS glue / bindings
└── matrixgame_rs.d.ts      ← TypeScript types (bonus)
```

---

## Step 10: Porting the MatrixGame Logic

Work through the C++ source module by module.  
Suggested order (easiest → hardest):

| Priority | C++ module | Rust equivalent | Notes |
|----------|-----------|-----------------|-------|
| 1 | Data structures (unit stats, map grid) | Plain Rust structs | Direct translation |
| 2 | Terrain rendering | `renderer/terrain.rs` + WGSL | Replace D3D calls with wgpu |
| 3 | Unit rendering | `renderer/units.rs` | Sprite batching or glTF models |
| 4 | Game loop / tick | `game/world.rs` | Remove Windows-specific timing |
| 5 | AI logic | `game/ai.rs` | Mostly pure computation, easy to port |
| 6 | Scripting | `game/script.rs` | Optional: add scripting if needed (no scripts in original engine) |
| 7 | Particle system | `renderer/particles.rs` | GPU instancing |
| 8 | Audio | `rodio` (native) / Web Audio API (WASM) | Platform-split |
| 9 | Multiplayer / network | `tokio` / WebSockets | Optional |

### Practical porting tips

- Keep a parallel build: compile both the old C++ and new Rust for the same
  test map and compare output pixel-by-pixel.
- Port one subsystem at a time; stub everything else.
- Use `#[cfg(debug_assertions)]` to add comparison assertions that you strip
  in release builds.

---

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| `std::time::Instant` panics on WASM | Use `crate::platform::now_secs()` |
| File I/O (`std::fs`) not available on WASM | Use `fetch` via `web-sys` |
| WASM binary is 20 MB | Enable `opt-level = "z"` and `lto = true` in `[profile.release]`; strip `image` crate features |
| WebGL 2 not supported on old iOS | Add a browser compatibility warning; target Safari 15+ |
| `.pkg` resource archives need unpacking | Write `assets/pkg_reader.rs` or pre-unpack offline and serve as static files |
| Threading (`std::thread`) not available on WASM | Use `wasm-bindgen-rayon` for parallel work, or avoid threads entirely |
| `async` runtime on WASM | Use `wasm-bindgen-futures::spawn_local`, not `tokio::spawn` |

---

## Debugging in the Browser

```javascript
// In DevTools console — the panic message will appear here thanks to
// console_error_panic_hook
```

```bash
# Build with debug info for better stack traces in the browser
wasm-pack build --dev --target web --out-dir pkg
```

Enable `RUST_LOG=debug` equivalent in WASM:
```rust
// src/lib.rs
console_log::init_with_level(log::Level::Debug).unwrap();
```

---

## Checklist

- [ ] `cargo build` succeeds for native target
- [ ] `wasm-pack build --target web` succeeds
- [ ] Blank window renders in browser (wgpu context working)
- [ ] Terrain mesh loads from unpacked SR2 assets
- [ ] Terrain texture renders correctly
- [ ] Camera controls work (mouse drag to rotate, scroll to zoom)
- [ ] At least one unit type renders on the map
- [ ] Game loop ticks at 60 fps in browser (check DevTools Performance tab)
- [ ] AI moves units correctly for one test scenario
- [ ] Full battle plays to completion matching original outcome

---

## Useful Resources

- [wgpu docs](https://docs.rs/wgpu) — main rendering API
- [wgpu examples](https://github.com/gfx-rs/wgpu/tree/trunk/examples) — start here for boilerplate
- [Learn wgpu tutorial](https://sotrh.github.io/learn-wgpu/) — best step-by-step guide
- [wasm-bindgen book](https://rustwasm.github.io/docs/wasm-bindgen/)
- [Bevy engine](https://bevyengine.org/) — if you want a full ECS framework instead of rolling your own
- [rhai scripting](https://rhai.rs/) — if you want to add a scripting layer (the original engine has no scripts)
- [MatrixGame source](https://github.com/twoweeks/MatrixGame) — the C++ original you're porting
