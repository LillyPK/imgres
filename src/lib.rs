//! GPU-accelerated resizing for in memory RGBA images.
//!
//! The crate accepts pixel buffers rather than decoding or encoding image
//! files. Create an [`Image`] with a nonzero [`Resolution`], then pass it to
//! [`resize`], [`resize_u8`], or [`resize_u16`] along with a `wgpu` device and
//! queue. Pixel data uses four channels per pixel in row major RGBA order.
//! Resizing uses bilinear filtering when either target dimension is larger
//! than its source dimension, and an area-weighted box filter otherwise.

use wgpu::util::DeviceExt;

/// Errors that can occur while constructing an image or resizing it.
#[derive(Debug)]
pub enum ResizeError {
    /// A resolution has a zero width or height.
    InvalidDimensions {
        /// The supplied width, which is zero.
        width: u32,
        /// The supplied height, which is zero.
        height: u32,
    },
    /// A buffer size could not be represented while performing this operation.
    BufferSizeOverflow {
        /// The calculation that overflowed.
        operation: &'static str,
    },
    /// The pixel buffer does not contain exactly four values per pixel.
    InvalidBufferLength {
        /// The image width in pixels.
        width: u32,
        /// The image height in pixels.
        height: u32,
        /// The required number of RGBA channel values.
        expected: usize,
        /// The actual number of values in the buffer.
        actual: usize,
    },
    /// The readback mapping callback did not deliver a result.
    ReadbackCallbackLost,
    /// The GPU rejected the request to map the readback buffer.
    ReadbackMapFailed {
        /// The underlying mapping error reported by `wgpu`.
        message: String,
    },
    /// Polling the GPU device failed while waiting for readback.
    DevicePollFailed {
        /// The underlying polling error reported by `wgpu`.
        message: String,
    },
    /// A GPU validation error occurred while setting up or submitting the resize.
    GpuValidationFailed {
        /// The validation error reported by `wgpu`.
        message: String,
    },
    /// Reserved for a request whose texture format is unsupported by the GPU.
    UnsupportedTextureFormat,
}

impl std::fmt::Display for ResizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => write!(
                f,
                "Invalid image dimensions: {width}×{height}. Width and height must both be greater than zero."
            ),
            Self::BufferSizeOverflow { operation } => write!(
                f,
                "The image buffer size overflowed while {operation}. Try smaller image dimensions."
            ),
            Self::InvalidBufferLength {
                width,
                height,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Invalid RGBA buffer length for a {width}×{height} image: expected {expected} channel values, got {actual}. RGBA requires four values per pixel; check that the buffer includes alpha values and matches the supplied dimensions."
                )?;
                let rgb_length = (*expected / 4).checked_mul(3);
                if rgb_length == Some(*actual) {
                    write!(
                        f,
                        " The buffer length matches three-channel RGB data; convert it to RGBA by adding an alpha value for every pixel."
                    )?;
                }
                Ok(())
            }
            Self::ReadbackCallbackLost => write!(
                f,
                "The GPU readback callback did not return a result. Retry the resize; if it persists, check whether the GPU device is still available."
            ),
            Self::ReadbackMapFailed { message } => write!(
                f,
                "Could not map the internal GPU readback buffer for CPU access: {message}. Retry the resize; if it persists, recreate the GPU device or report this error."
            ),
            Self::DevicePollFailed { message } => write!(
                f,
                "Failed while waiting for GPU readback: {message}. Check that the GPU device is still available and retry the operation."
            ),
            Self::GpuValidationFailed { message } => write!(
                f,
                "The GPU rejected the resize operation during validation: {message}. Check the requested dimensions and device format support; if those are valid, report this as a possible internal GPU setup issue."
            ),
            Self::UnsupportedTextureFormat => write!(
                f,
                "The GPU does not support the texture format required for this pixel type. Check the device's format capabilities."
            ),
        }
    }
}

impl std::error::Error for ResizeError {}

/// The width and height, in pixels, of an image.
///
/// Use [`Resolution::new`] to ensure both dimensions are nonzero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolution {
    /// The image width in pixels.
    pub width: u32,
    /// The image height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Creates a resolution with nonzero dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ResizeError::InvalidDimensions`] if either dimension is zero.
    pub fn new(width: u32, height: u32) -> Result<Self, ResizeError> {
        if width == 0 || height == 0 {
            return Err(ResizeError::InvalidDimensions { width, height });
        }
        Ok(Self { width, height })
    }

    fn pixel_count(self) -> Result<usize, ResizeError> {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .ok_or(ResizeError::BufferSizeOverflow {
                operation: "calculating the pixel count",
            })
    }
}

fn rgba_buffer_len(resolution: Resolution) -> Result<usize, ResizeError> {
    if resolution.width == 0 || resolution.height == 0 {
        return Err(ResizeError::InvalidDimensions {
            width: resolution.width,
            height: resolution.height,
        });
    }

    resolution
        .pixel_count()?
        .checked_mul(4)
        .ok_or(ResizeError::BufferSizeOverflow {
            operation: "calculating the RGBA channel count",
        })
}

/// An image resolution and its owned pixel buffer.
///
/// The buffer contains four channel values per pixel in row-major RGBA order.
/// For example, an image with resolution `width` by `height` must contain exactly
/// `width * height * 4` values.
#[derive(Debug)]
pub struct Image<T> {
    /// The dimensions of this image.
    pub resolution: Resolution,
    pixels: Vec<T>,
}

impl<T> Image<T> {
    /// Creates an image from its resolution and RGBA pixel data.
    ///
    /// # Errors
    ///
    /// Returns [`ResizeError::InvalidDimensions`] if either dimension is zero,
    /// [`ResizeError::BufferSizeOverflow`] if the expected buffer length cannot
    /// be represented, or [`ResizeError::InvalidBufferLength`] if `pixels` does
    /// not contain exactly four values per pixel.
    pub fn new(resolution: Resolution, pixels: Vec<T>) -> Result<Self, ResizeError> {
        let expected = rgba_buffer_len(resolution)?;

        if pixels.len() != expected {
            return Err(ResizeError::InvalidBufferLength {
                width: resolution.width,
                height: resolution.height,
                expected,
                actual: pixels.len(),
            });
        }

        Ok(Self { resolution, pixels })
    }

    /// Borrows the row-major RGBA pixel data.
    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }

    /// Consumes the image and returns its owned pixel buffer.
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

/// Describes the GPU texture representation and shader used by [`resize`].
///
/// The crate implements this trait for `u8` and `u16`. Custom implementations
/// must use a texture format and shader compatible with the RGBA texture
/// bindings and output representation expected by the resizing pipeline.
pub trait GpuPixel: bytemuck::Pod + Copy {
    /// The `wgpu` texture format used to store each pixel.
    const FORMAT: wgpu::TextureFormat;
    /// The number of bytes occupied by one four-channel pixel.
    const BYTES_PER_PIXEL: u32;

    /// Returns the WGSL shader source for the requested resize direction.
    ///
    /// `upscale` is true when at least one target dimension exceeds the
    /// corresponding source dimension.
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

/// Resizes an image using the GPU.
///
/// The input and output contain four channel values per pixel in row-major
/// RGBA order. Bilinear filtering is selected if either target dimension is
/// larger than its corresponding source dimension; otherwise, resizing uses
/// an area-weighted box filter. The returned image owns its CPU-side pixel
/// buffer.
///
/// # Errors
///
/// Returns [`ResizeError::InvalidDimensions`] for zero-sized input or target
/// dimensions, [`ResizeError::BufferSizeOverflow`] if a required buffer size
/// overflows, [`ResizeError::InvalidBufferLength`] if the input pixel count
/// does not match its resolution, or a readback or GPU validation error if the
/// GPU operation cannot be completed.
pub async fn resize<T: GpuPixel>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Image<T>,
    target: Resolution,
) -> Result<Image<T>, ResizeError> {
    let expected = rgba_buffer_len(input.resolution)?;

    if input.pixels.len() != expected {
        return Err(ResizeError::InvalidBufferLength {
            width: input.resolution.width,
            height: input.resolution.height,
            expected,
            actual: input.pixels.len(),
        });
    }

    let output_len = rgba_buffer_len(target)?;
    let output_byte_len = output_len.checked_mul(std::mem::size_of::<T>()).ok_or(
        ResizeError::BufferSizeOverflow {
            operation: "calculating the output byte length",
        },
    )?;
    let source_bytes_per_row = input
        .resolution
        .width
        .checked_mul(T::BYTES_PER_PIXEL)
        .ok_or(ResizeError::BufferSizeOverflow {
            operation: "calculating the source row byte length",
        })?;
    let unpadded_bytes_per_row =
        target
            .width
            .checked_mul(T::BYTES_PER_PIXEL)
            .ok_or(ResizeError::BufferSizeOverflow {
                operation: "calculating the destination row byte length",
            })?;
    let padded_bytes_per_row = unpadded_bytes_per_row
        .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .ok_or(ResizeError::BufferSizeOverflow {
            operation: "aligning the destination row byte length",
        })?;
    let readback_size = (padded_bytes_per_row as u64)
        .checked_mul(target.height as u64)
        .ok_or(ResizeError::BufferSizeOverflow {
            operation: "calculating the GPU readback buffer size",
        })?;

    let upscale = target.width > input.resolution.width || target.height > input.resolution.height;

    let format = T::FORMAT;
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

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
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
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
            bytes_per_row: Some(source_bytes_per_row),
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
        source: wgpu::ShaderSource::Wgsl(T::shader(upscale).into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resize-compute-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let uniforms = Uniforms {
        src_size: [
            input.resolution.width as f32,
            input.resolution.height as f32,
        ],
        dst_size: [target.width as f32, target.height as f32],
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("resize-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let src_view = src.create_view(&Default::default());
    let dst_view = dst.create_view(&Default::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resize-readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("resize-command-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resize-compute-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(target.width.div_ceil(16), target.height.div_ceil(16), 1);
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

    if let Some(error) = error_scope.pop().await {
        return Err(ResizeError::GpuValidationFailed {
            message: error.to_string(),
        });
    }

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();

    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| ResizeError::DevicePollFailed {
            message: error.to_string(),
        })?;

    receiver
        .recv()
        .map_err(|_| ResizeError::ReadbackCallbackLost)?
        .map_err(|error| ResizeError::ReadbackMapFailed {
            message: error.to_string(),
        })?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|error| ResizeError::ReadbackMapFailed {
            message: error.to_string(),
        })?;
    let row_size = unpadded_bytes_per_row as usize;
    let padded_size = padded_bytes_per_row as usize;

    let mut bytes = Vec::with_capacity(output_byte_len);

    for y in 0..target.height as usize {
        let start = y * padded_size;
        bytes.extend_from_slice(&mapped[start..start + row_size]);
    }

    drop(mapped);
    readback.unmap();

    debug_assert_eq!(bytes.len(), output_byte_len);

    let pixels: Vec<T> = bytemuck::cast_slice(&bytes).to_vec();

    Image::new(target, pixels)
}

/// Resizes an 8-bit-per-channel RGBA image using the GPU.
///
/// This is the `u8` convenience wrapper around [`resize`]. Its input and
/// output use four `u8` channel values per pixel in row-major RGBA order.
///
/// # Errors
///
/// Returns the errors documented by [`resize`].
pub async fn resize_u8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Image<u8>,
    target: Resolution,
) -> Result<Image<u8>, ResizeError> {
    resize(device, queue, input, target).await
}

/// Resizes a 16-bit-per-channel RGBA image using the GPU.
///
/// This is the `u16` convenience wrapper around [`resize`]. Its input and
/// output use four `u16` channel values per pixel in row-major RGBA order.
///
/// # Errors
///
/// Returns the errors documented by [`resize`].
pub async fn resize_u16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Image<u16>,
    target: Resolution,
) -> Result<Image<u16>, ResizeError> {
    resize(device, queue, input, target).await
}
