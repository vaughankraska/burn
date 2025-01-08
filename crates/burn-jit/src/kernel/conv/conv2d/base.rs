use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

use crate::{
    kernel::conv::tune::conv2d_autotune, ops::numeric::empty_device, tensor::JitTensor,
    FloatElement, JitRuntime,
};

use super::{gemm::conv2d_gemm_cmma_large_m, im2col::conv2d_im2col};

// #[cfg(feature = "autotune")]
// use super::tune::conv2d_autotune;

/// The strategy to be used when launching a convolution kernel.
pub enum Conv2dStrategy {
    /// A simple direct convolution.
    Direct,
    /// GEMM (im2col) based implementation of convolution. Significantly increased memory usage.
    Gemm,
    /// Implicit GEMM implementation of convolution. Lower memory usage but requires CMMA and
    /// has constraints on tensor shape.
    ImplicitGemm,
    /// Implicit GEMM implementation of convolution. Uses `cubecl` matmul components to provide
    /// the flexibility needed to work well for varied problem sizes.
    ImplicitGemmComplex,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
}

impl Default for Conv2dStrategy {
    fn default() -> Self {
        #[cfg(feature = "autotune")]
        return Conv2dStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        Conv2dStrategy::ImplicitGemmComplex
    }
}

/// Perform a 2D convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
///
pub fn conv2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
    strategy: Conv2dStrategy,
) -> JitTensor<R> {
    match strategy {
        Conv2dStrategy::Direct => {
            let client = &input.client;

            let out = empty_device::<R, E>(
                client.clone(),
                input.device.clone(),
                shape_out(&input, &weight, &options),
            );

            cubecl::convolution::conv2d::direct::conv2d_direct::<R, E>(
                client,
                input.as_handle_ref(),
                weight.as_handle_ref(),
                bias.as_ref().map(|bias_ref| bias_ref.as_handle_ref()),
                out.as_handle_ref(),
                options.into(),
            );

            out
        }
        #[cfg(feature = "autotune")]
        Conv2dStrategy::Autotune => conv2d_autotune::<R, E>(input, weight, bias, options),
        Conv2dStrategy::Gemm => conv2d_im2col::<R, E>(input, weight, bias, options),
        // Conv2dStrategy::ImplicitGemm => conv2d_implicit_gemm::<R, E>(input, weight, bias, options),
        Conv2dStrategy::ImplicitGemm => todo!(),
        Conv2dStrategy::ImplicitGemmComplex => {
            conv2d_gemm_cmma_large_m::<R, E>(input, weight, bias, options)
        }
    }
}

fn shape_out<R: JitRuntime>(
    input: &JitTensor<R>,
    weight: &JitTensor<R>,
    options: &ConvOptions<2>,
) -> Shape {
    let [batch_size, _, in_height, in_width] = input.shape.dims[0..4]
        .try_into()
        .expect("Input shape should have 4 dimensions");

    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims[0..4]
        .try_into()
        .expect("Weight shape should have 4 dimensions");

    let out_h = calculate_conv_output_size(
        kernel_h,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    Shape::new([batch_size, out_channels, out_h, out_w])
}

/// Custom transpose kernel for specific shape swap
pub fn nchw_to_nhwc<R: JitRuntime, E: FloatElement>(input: JitTensor<R>) -> JitTensor<R> {
    let [batch_size, in_c, h, w] = input
        .shape
        .dims
        .clone()
        .try_into()
        .expect("Input shape should have 4 dimensions");
    let out_shape = Shape::new([batch_size, h, w, in_c]);
    let output = empty_device::<R, E>(input.client.clone(), input.device.clone(), out_shape);

    cubecl::convolution::conv2d::nchw_to_nhwc::<R, E>(
        &input.client,
        input.as_handle_ref(),
        output.as_handle_ref(),
    );

    output
}
