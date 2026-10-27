use imgres::{Image, ResizeError, Resolution};

#[test]
fn accepts_correct_rgba_buffer_lengths_for_u8_and_u16() {
    let resolution = Resolution::new(2, 2).unwrap();

    let image_u8 = Image::new(resolution, vec![0_u8; 2 * 2 * 4]).unwrap();
    assert_eq!(image_u8.pixels().len(), 16);
    assert_eq!(image_u8.resolution, resolution);

    let image_u16 = Image::new(resolution, vec![0_u16; 2 * 2 * 4]).unwrap();
    assert_eq!(image_u16.pixels().len(), 16);
    assert_eq!(image_u16.resolution, resolution);
}

#[test]
fn invalid_rgb_length_reports_expected_values_and_conversion_hint() {
    let resolution = Resolution::new(2, 2).unwrap();
    let error = Image::new(resolution, vec![0_u8; 2 * 2 * 3]).unwrap_err();

    assert!(matches!(
        error,
        ResizeError::InvalidBufferLength {
            width: 2,
            height: 2,
            expected: 16,
            actual: 12,
        }
    ));
    assert_eq!(
        error.to_string(),
        "Invalid RGBA buffer length for a 2×2 image: expected 16 channel values, got 12. RGBA requires four values per pixel; check that the buffer includes alpha values and matches the supplied dimensions. The buffer length matches three-channel RGB data; convert it to RGBA by adding an alpha value for every pixel."
    );
}

#[test]
fn invalid_non_rgb_length_reports_actual_and_expected_values() {
    let resolution = Resolution::new(2, 2).unwrap();
    let error = Image::new(resolution, vec![0_u8; 10]).unwrap_err();

    assert!(matches!(
        error,
        ResizeError::InvalidBufferLength {
            width: 2,
            height: 2,
            expected: 16,
            actual: 10,
        }
    ));
    assert_eq!(
        error.to_string(),
        "Invalid RGBA buffer length for a 2×2 image: expected 16 channel values, got 10. RGBA requires four values per pixel; check that the buffer includes alpha values and matches the supplied dimensions."
    );
}

#[test]
fn rejects_zero_dimensions_even_when_resolution_is_constructed_directly() {
    let error = Image::<u8>::new(
        Resolution {
            width: 0,
            height: 2,
        },
        Vec::new(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        ResizeError::InvalidDimensions {
            width: 0,
            height: 2,
        }
    ));
    assert_eq!(
        error.to_string(),
        "Invalid image dimensions: 0×2. Width and height must both be greater than zero."
    );
}

#[test]
fn resolution_constructor_reports_zero_dimensions() {
    let error = Resolution::new(3, 0).unwrap_err();

    assert!(matches!(
        error,
        ResizeError::InvalidDimensions {
            width: 3,
            height: 0,
        }
    ));
    assert_eq!(
        error.to_string(),
        "Invalid image dimensions: 3×0. Width and height must both be greater than zero."
    );
}

#[test]
fn readback_errors_include_the_underlying_error_and_user_hint() {
    let map_error = ResizeError::ReadbackMapFailed {
        message: "mapping rejected".to_owned(),
    };
    assert_eq!(
        map_error.to_string(),
        "Could not map the internal GPU readback buffer for CPU access: mapping rejected. Retry the resize; if it persists, recreate the GPU device or report this error."
    );

    assert_eq!(
        ResizeError::ReadbackCallbackLost.to_string(),
        "The GPU readback callback did not return a result. Retry the resize; if it persists, check whether the GPU device is still available."
    );
}
