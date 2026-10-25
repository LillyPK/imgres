
use imgres::{resize, Image, Resolution};
use std::{
    error::Error,
    path::PathBuf,
    sync::Arc,
};

use softbuffer::{Context, Surface};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct Viewer {
    image: Image<u8>,
    device: wgpu::Device,
    queue: wgpu::Queue,

    window: Option<Arc<Window>>,
    context: Option<Context<Arc<Window>>>,
    surface: Option<Surface<Arc<Window>, Arc<Window>>>,

    last_size: Option<PhysicalSize<u32>>,
    pixels: Vec<u32>,
}

impl Viewer {
    fn new(
        image: Image<u8>,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self {
            image,
            device,
            queue,
            window: None,
            context: None,
            surface: None,
            last_size: None,
            pixels: Vec::new(),
        }
    }

    fn redraw(&mut self) -> Result<(), Box<dyn Error>> {
        let Some(window) = self.window.as_ref() else {
            return Ok(());
        };

        let size = window.inner_size();

        // A minimized window can have a zero-sized client area.
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        // Only resize the image when the client dimensions change.
        if self.last_size != Some(size) {
            let target = Resolution::new(size.width, size.height)?;

            let resized = pollster::block_on(resize::<u8>(
                &self.device,
                &self.queue,
                &self.image,
                target,
            ))?;

            // softbuffer uses packed 0x00RRGGBB pixels.
            // Alpha is omitted for this basic viewer.
            self.pixels = resized
                .pixels()
                .chunks_exact(4)
                .map(|p| {
                    ((p[0] as u32) << 16)
                        | ((p[1] as u32) << 8)
                        | (p[2] as u32)
                })
                .collect();

            self.last_size = Some(size);
        }

        let surface = self
            .surface
            .as_mut()
            .ok_or("Window surface is not initialized")?;

        surface
            .resize(
                std::num::NonZeroU32::new(size.width)
                    .ok_or("Zero window width")?,
                std::num::NonZeroU32::new(size.height)
                    .ok_or("Zero window height")?,
            )?;

        let mut buffer = surface.buffer_mut()?;

        if buffer.len() != self.pixels.len() {
            return Err("Display buffer dimensions do not match".into());
        }

        buffer.copy_from_slice(&self.pixels);
        buffer.present()?;

        Ok(())
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("imgres viewer")
            .with_inner_size(PhysicalSize::new(800, 600))
            .with_resizable(true);

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("Failed to create window"),
        );

        let context = Context::new(window.clone())
            .expect("Failed to create softbuffer context");

        let surface = Surface::new(&context, window.clone())
            .expect("Failed to create softbuffer surface");

        self.window = Some(window.clone());
        self.context = Some(context);
        self.surface = Some(surface);

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    // Force a new resize on the next redraw.
                    self.last_size = None;
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    eprintln!("Viewer error: {error}");
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("Usage: viewer.exe <image-path>")?;

    // Decode the input image into raw RGBA8 pixels.
    let decoded = image::open(&path)?.to_rgba8();

    let resolution = Resolution::new(
        decoded.width(),
        decoded.height(),
    )?;

    let image = Image::new(
        resolution,
        decoded.into_raw(),
    )?;

    // This crate only needs a compute-capable adapter.
    let instance = wgpu::Instance::default();

    let adapter = pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        },
    ))?;

    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("imgres viewer device"),
            ..Default::default()
        }),
    )?;

    let event_loop = EventLoop::new()?;
    let mut viewer = Viewer::new(image, device, queue);

    event_loop.run_app(&mut viewer)?;

    Ok(())
}
