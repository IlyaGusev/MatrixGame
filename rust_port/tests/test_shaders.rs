//! Validates every WGSL file in `shaders/` on a headless device, so
//! shader syntax/typing errors surface in `cargo test` instead of at
//! first use in the browser.

#[test]
fn all_wgsl_shaders_compile() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        flags: wgpu::InstanceFlags::empty(),
        ..Default::default()
    });
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok() else {
        eprintln!("no adapter available; skipping shader validation");
        return;
    };
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device");

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).expect("shaders dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read shader");
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: path.file_name().and_then(|n| n.to_str()),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "{}: {}", path.display(), err.unwrap());
        checked += 1;
    }
    assert!(checked > 0, "no shaders found in {}", dir.display());
}
