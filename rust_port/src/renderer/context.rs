use std::sync::Arc;
use winit::window::Window;

pub struct GfxContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl GfxContext {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        // Arc<Window> gives us Surface<'static>
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("matrixgame device"),
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("failed to create device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1).min(device.limits().max_texture_dimension_2d),
            height: size.height.max(1).min(device.limits().max_texture_dimension_2d),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let max = self.device.limits().max_texture_dimension_2d;
        self.config.width = width.max(1).min(max);
        self.config.height = height.max(1).min(max);
        self.surface.configure(&self.device, &self.config);
    }

    pub fn begin_frame(&self) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView, wgpu::CommandEncoder), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        Ok((output, view, encoder))
    }

    pub fn end_frame(&self, output: wgpu::SurfaceTexture, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
