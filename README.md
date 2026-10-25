# imgres

`imgres` resizes raw RGBA pixel buffers using GPU compute shaders with [`wgpu`](https://wgpu.rs/).

It accepts pixel data in memory rather than loading or saving image files. Your application is responsible for decoding images into RGBA pixels and encoding or displaying the result.

## Features

- GPU-accelerated resizing through `wgpu`.
- Supports 8-bit (`u8`) and 16-bit (`u16`) channel data.
- Expects four channels per pixel, in RGBA order.
- Uses bilinear filtering when upscaling and an area-weighted box filter when downscaling.
- Returns the resized pixels in a CPU-side `Vec`.

## Installation

Add `imgres` to your `Cargo.toml`:

```toml
[dependencies]
imgres = "0.1.0"
wgpu = "30.0.1"
```

If you are using a local checkout, use a path dependency instead:

```toml
[dependencies]
imgres = { path = "../imgres" }
wgpu = "30.0.1"
```

## Example

The following example resizes an RGBA `u8` image to `640 × 480`. It assumes your application has already created a `wgpu::Device` and `wgpu::Queue`.

```rust
use imgres::{resize_u8, Image, Resolution};

async fn resize_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba_pixels: Vec<u8>,
    source_width: u32,
    source_height: u32,
) -> Result<Image<u8>, Box<dyn std::error::Error>> {
    // The buffer must contain exactly width * height * 4 channel values.
    let source_resolution = Resolution::new(source_width, source_height)?;
    let source_image = Image::new(source_resolution, rgba_pixels)?;

    let target_resolution = Resolution::new(640, 480)?;
    let resized = resize_u8(
        device,
        queue,
        &source_image,
        target_resolution,
    )
    .await?;

    Ok(resized)
}
```

Read the result using `Image::pixels()`:

```rust
let rgba_pixels: &[u8] = resized.pixels();
```

The returned slice contains four values per pixel in row-major RGBA order. Use `Image::into_pixels()` if you want to take ownership of the output `Vec`.

For 16-bit channel data, use `Image<u16>` and `resize_u16` instead.

## API overview

### `Resolution`

Represents an image's width and height.

```rust
let resolution = Resolution::new(1920, 1080)?;
```

Both dimensions must be greater than zero. Creating a resolution with a zero width or height returns `ResizeError::InvalidDimensions`.

### `Image<T>`

Owns an image's pixel buffer and its resolution.

```rust
let image = Image::new(resolution, pixels)?;
```

The pixel buffer must contain exactly `width * height * 4` values. An invalid buffer length returns `ResizeError::InvalidBufferLength`. Use:

- `image.resolution` to read its `Resolution`.
- `image.pixels()` to borrow the pixel data.
- `image.into_pixels()` to consume the image and retrieve its `Vec<T>`.

For `u8`, each channel is an 8-bit value. For `u16`, each channel is a 16-bit unsigned integer. In either case, data is expected to be four channels per pixel in RGBA order.

### Resize functions

Both functions are asynchronous and require a `wgpu::Device` and `wgpu::Queue` supplied by the calling application.

| Function | Input and output | Description |
|---|---|---|
| `resize_u8` | `Image<u8>` | Resizes 8-bit RGBA channel data. |
| `resize_u16` | `Image<u16>` | Resizes 16-bit RGBA channel data. |

Each function takes the device, queue, a reference to the input image, and the target `Resolution`. It returns `Result<Image<T>, ResizeError>`.

The current filter selection is based on the target dimensions: if either target dimension is larger than its source dimension, the bilinear shader is used; otherwise, the area-weighted box shader is used.

### `ResizeError`

Possible errors include:

- `InvalidDimensions` — a resolution has a zero width or height.
- `BufferSizeOverflow` — calculating the required buffer size overflowed.
- `InvalidBufferLength` — the pixel buffer does not match the image dimensions.
- `MapFailed` — the GPU readback buffer could not be mapped.
- `UnsupportedTextureFormat` — the requested texture format is unsupported.

## Viewer example

The repository's `examples/viewer.rs` example is a separate test viewer. If its development dependencies are installed, run it with an image path:

```sh
cargo run --example viewer -- path/to/image.png
```

The viewer's image-loading and window-display dependencies are for the example only; `imgres` itself operates on raw pixel buffers.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
