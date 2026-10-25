use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum ResizeError {
    InvalidDimensions,
    BufferSizeOverflow,
    InvalidBufferLength,
    MapFailed,
    UnsupportedTextureFormat,
}

impl std::fmt::Display for ResizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ResizeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub fn new(width: u32, height: u32) -> Result<Self, ResizeError> {
        if width == 0 || height == 0 {
            return Err(ResizeError::InvalidDimensions);
        }
        Ok(Self { width, height })
    }

    fn pixel_count(self) -> Result<usize, ResizeError> {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .ok_or(ResizeError::BufferSizeOverflow)
    }
}

#[derive(Debug)]
pub struct Image<T> {
    pub resolution: Resolution,
    pixels: Vec<T>,
}

impl<T> Image<T> {
    pub fn new(
        resolution: Resolution,
        pixels: Vec<T>,
    ) -> Result<Self, ResizeError> {
        let expected = resolution
            .pixel_count()?
            .checked_mul(4)
            .ok_or(ResizeError::BufferSizeOverflow)?;

        if pixels.len() != expected {
            return Err(ResizeError::InvalidBufferLength);
        }

        Ok(Self { resolution, pixels })
    }

    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }

    pub fn into_pixels(self) -> Vec<T> {
        self.pixels
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    src_size: [f32; 2],
    dst_size: [f32; 2],
}

const BILINEAR_U8: &str = r#"
@group(0) @binding(0)
var src_texture: texture_2d<f32>;

@group(0) @binding(1)
var dst_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    src_size: vec2<f32>,
    dst_size: vec2<f32>,
};

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = gid.xy;

    if (dst.x >= u32(uniforms.dst_size.x) ||
        dst.y >= u32(uniforms.dst_size.y)) {
        return;
    }

    let scale = uniforms.src_size / uniforms.dst_size;
    let p = (vec2<f32>(dst) + vec2<f32>(0.5)) * scale
        - vec2<f32>(0.5);

    let base = vec2<i32>(floor(p));
    let weight = fract(p);
    let max_coord = vec2<i32>(uniforms.src_size) - vec2<i32>(1);

    let x0 = clamp(base.x, 0, max_coord.x);
    let y0 = clamp(base.y, 0, max_coord.y);
    let x1 = clamp(base.x + 1, 0, max_coord.x);
    let y1 = clamp(base.y + 1, 0, max_coord.y);

    let p00 = textureLoad(src_texture, vec2<i32>(x0, y0), 0);
    let p10 = textureLoad(src_texture, vec2<i32>(x1, y0), 0);
    let p01 = textureLoad(src_texture, vec2<i32>(x0, y1), 0);
    let p11 = textureLoad(src_texture, vec2<i32>(x1, y1), 0);

    let top = mix(p00, p10, weight.x);
    let bottom = mix(p01, p11, weight.x);

    textureStore(
        dst_texture,
        vec2<i32>(dst),
        mix(top, bottom, weight.y)
    );
}
"#;

const BILINEAR_U16: &str = r#"
@group(0) @binding(0)
var src_texture: texture_2d<u32>;

@group(0) @binding(1)
var dst_texture: texture_storage_2d<rgba16uint, write>;

struct Uniforms {
    src_size: vec2<f32>,
    dst_size: vec2<f32>,
};

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

fn load_pixel(p: vec2<i32>) -> vec4<f32> {
    return vec4<f32>(textureLoad(src_texture, p, 0));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = gid.xy;

    if (dst.x >= u32(uniforms.dst_size.x) ||
        dst.y >= u32(uniforms.dst_size.y)) {
        return;
    }

    let scale = uniforms.src_size / uniforms.dst_size;
    let p = (vec2<f32>(dst) + vec2<f32>(0.5)) * scale
        - vec2<f32>(0.5);

    let base = vec2<i32>(floor(p));
    let weight = fract(p);
    let max_coord = vec2<i32>(uniforms.src_size) - vec2<i32>(1);

    let x0 = clamp(base.x, 0, max_coord.x);
    let y0 = clamp(base.y, 0, max_coord.y);
    let x1 = clamp(base.x + 1, 0, max_coord.x);
    let y1 = clamp(base.y + 1, 0, max_coord.y);

    let p00 = load_pixel(vec2<i32>(x0, y0));
    let p10 = load_pixel(vec2<i32>(x1, y0));
    let p01 = load_pixel(vec2<i32>(x0, y1));
    let p11 = load_pixel(vec2<i32>(x1, y1));

    let top = mix(p00, p10, weight.x);
    let bottom = mix(p01, p11, weight.x);
    let value = mix(top, bottom, weight.y);

    textureStore(
        dst_texture,
        vec2<i32>(dst),
        vec4<u32>(round(clamp(value, vec4<f32>(0.0),
            vec4<f32>(65535.0))))
    );
}
"#;

// Area-weighted box filter. Unlike a fixed 2x2 average, this
// supports arbitrary downscaling ratios.
const BOX_U8: &str = r#"
@group(0) @binding(0)
var src_texture: texture_2d<f32>;

@group(0) @binding(1)
var dst_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    src_size: vec2<f32>,
    dst_size: vec2<f32>,
};

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = gid.xy;

    if (dst.x >= u32(uniforms.dst_size.x) ||
        dst.y >= u32(uniforms.dst_size.y)) {
        return;
    }

    let scale = uniforms.src_size / uniforms.dst_size;
    let lo = vec2<f32>(dst) * scale;
    let hi = vec2<f32>(dst + vec2<u32>(1u)) * scale;

    let first = vec2<i32>(floor(lo));
    let last = vec2<i32>(ceil(hi)) - vec2<i32>(1);
    let src_max = vec2<i32>(uniforms.src_size) - vec2<i32>(1);

    var total = vec4<f32>(0.0);
    var total_weight = 0.0;

    for (var y = first.y; y <= last.y; y = y + 1) {
        for (var x = first.x; x <= last.x; x = x + 1) {
            let pixel_lo = vec2<f32>(f32(x), f32(y));
            let pixel_hi = pixel_lo + vec2<f32>(1.0);

            let overlap = max(
                vec2<f32>(0.0),
                min(hi, pixel_hi) - max(lo, pixel_lo)
            );

            let weight = overlap.x * overlap.y;
            let coord = clamp(vec2<i32>(x, y),
                vec2<i32>(0), src_max);

            total += textureLoad(src_texture, coord, 0) * weight;
            total_weight += weight;
        }
    }

    textureStore(
        dst_texture,
        vec2<i32>(dst),
        total / total_weight
    );
}
"#;

const BOX_U16: &str = r#"
@group(0) @binding(0)
var src_texture: texture_2d<u32>;

@group(0) @binding(1)
var dst_texture: texture_storage_2d<rgba16uint, write>;

struct Uniforms {
    src_size: vec2<f32>,
    dst_size: vec2<f32>,
};

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = gid.xy;

    if (dst.x >= u32(uniforms.dst_size.x) ||
        dst.y >= u32(uniforms.dst_size.y)) {
        return;
    }

    let scale = uniforms.src_size / uniforms.dst_size;
    let lo = vec2<f32>(dst) * scale;
    let hi = vec2<f32>(dst + vec2<u32>(1u)) * scale;

    let first = vec2<i32>(floor(lo));
    let last = vec2<i32>(ceil(hi)) - vec2<i32>(1);
    let src_max = vec2<i32>(uniforms.src_size) - vec2<i32>(1);

    var total = vec4<f32>(0.0);
    var total_weight = 0.0;

    for (var y = first.y; y <= last.y; y = y + 1) {
        for (var x = first.x; x <= last.x; x = x + 1) {
            let pixel_lo = vec2<f32>(f32(x), f32(y));
            let pixel_hi = pixel_lo + vec2<f32>(1.0);

            let overlap = max(
                vec2<f32>(0.0),
                min(hi, pixel_hi) - max(lo, pixel_lo)
            );

            let weight = overlap.x * overlap.y;
            let coord = clamp(vec2<i32>(x, y),
                vec2<i32>(0), src_max);

            total += vec4<f32>(
                textureLoad(src_texture, coord, 0)
            ) * weight;

            total_weight += weight;
        }
    }

    let value = total / total_weight;

    textureStore(
        dst_texture,
        vec2<i32>(dst),
        vec4<u32>(round(clamp(value,
            vec4<f32>(0.0), vec4<f32>(65535.0))))
    );
}
"#;

trait GpuPixel: bytemuck::Pod + Copy {
    const FORMAT: wgpu::TextureFormat;
    const BYTES_PER_PIXEL: u32;

    fn shader(upscale: bool) -> &'static str;
}

impl GpuPixel for u8 {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const BYTES_PER_PIXEL: u32 = 4;

    fn shader(upscale: bool) -> &'static str {
        if upscale { BILINEAR_U8 } else { BOX_U8 }
    }
}

impl GpuPixel for u16 {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Uint;
    const BYTES_PER_PIXEL: u32 = 8;

    fn shader(upscale: bool) -> &'static str {
        if upscale { BILINEAR_U16 } else { BOX_U16 }
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

pub async fn resize<T: GpuPixel>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Image<T>,
    target: Resolution,
) -> Result<Image<T>, ResizeError> {
    let expected = input.resolution.pixel_count()?
        .checked_mul(4)
        .ok_or(ResizeError::BufferSizeOverflow)?;

    if input.pixels.len() != expected {
        return Err(ResizeError::InvalidBufferLength);
    }

    let output_len = target.pixel_count()?
        .checked_mul(4)
        .ok_or(ResizeError::BufferSizeOverflow)?;

    let upscale = target.width > input.resolution.width
        || target.height > input.resolution.height;

    let format = T::FORMAT;

    let src = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resize-source"),
        size: wgpu::Extent3d {
            width: input.resolution.width,
            height: input.resolution.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let dst = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resize-destination"),
        size: wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&input.pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(
                input.resolution.width * T::BYTES_PER_PIXEL
            ),
            rows_per_image: Some(input.resolution.height),
        },
        wgpu::Extent3d {
            width: input.resolution.width,
            height: input.resolution.height,
            depth_or_array_layers: 1,
        },
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resize-compute-shader"),
        source: wgpu::ShaderSource::Wgsl(
            T::shader(upscale).into()
        ),
    });

    let pipeline = device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some("resize-compute-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        },
    );

    let uniforms = Uniforms {
        src_size: [
            input.resolution.width as f32,
            input.resolution.height as f32,
        ],
        dst_size: [target.width as f32, target.height as f32],
    };

    let uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("resize-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );

    let src_view = src.create_view(&Default::default());
    let dst_view = dst.create_view(&Default::default());

    let bind_group = device.create_bind_group(
        &wgpu::BindGroupDescriptor {
            label: Some("resize-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        },
    );

    let unpadded_bytes_per_row = target.width
        .checked_mul(T::BYTES_PER_PIXEL)
        .ok_or(ResizeError::BufferSizeOverflow)?;

    let padded_bytes_per_row = align_to(
        unpadded_bytes_per_row,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    );

    let readback_size = (padded_bytes_per_row as u64)
        .checked_mul(target.height as u64)
        .ok_or(ResizeError::BufferSizeOverflow)?;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resize-readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("resize-command-encoder"),
        },
    );

    {
        let mut pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some("resize-compute-pass"),
                timestamp_writes: None,
            },
        );

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            target.width.div_ceil(16),
            target.height.div_ceil(16),
            1,
        );
    }

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(target.height),
            },
        },
        wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();

    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();


    receiver.recv()
        .map_err(|_| ResizeError::MapFailed)?
        .map_err(|_| ResizeError::MapFailed)?;

    let mapped = slice
    .get_mapped_range()
    .map_err(|_| ResizeError::MapFailed)?;
    let bytes_per_pixel = T::BYTES_PER_PIXEL as usize;
    let row_size = unpadded_bytes_per_row as usize;
    let padded_size = padded_bytes_per_row as usize;

    let mut bytes = Vec::with_capacity(
        output_len * std::mem::size_of::<T>()
    );

    for y in 0..target.height as usize {
        let start = y * padded_size;
        bytes.extend_from_slice(
            &mapped[start..start + row_size]
        );
    }

    drop(mapped);
    readback.unmap();

    debug_assert_eq!(
        bytes.len(),
        output_len * std::mem::size_of::<T>()
    );
    let _ = bytes_per_pixel;

    let pixels: Vec<T> = bytemuck::cast_slice(&bytes).to_vec();

    Image::new(target, pixels)
}

pub async fn resize_u8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Image<u8>,
    target: Resolution,
) -> Result<Image<u8>, ResizeError> {
    resize(device, queue, input, target).await
}

pub async fn resize_u16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Image<u16>,
    target: Resolution,
) -> Result<Image<u16>, ResizeError> {
    resize(device, queue, input, target).await
}
