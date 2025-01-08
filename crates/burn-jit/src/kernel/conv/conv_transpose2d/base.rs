use burn_tensor::{
    ops::{conv::calculate_conv_transpose_output_size, ConvTransposeOptions},
    Shape,
};

use crate::{
    kernel::conv::conv_transpose2d::col2im, ops::numeric::empty_device, tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[cfg(feature = "autotune")]
use crate::kernel::conv::tune::conv_transpose2d_autotune;

/// The strategy to be used when launching a conv_transpose kernel.
pub enum ConvTranspose2dStrategy {
    /// A simple direct convolution.
    Direct,
    /// GEMM (im2col) based implementation of convolution. Significantly increased memory usage.
    Gemm,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
}

impl Default for ConvTranspose2dStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return ConvTranspose2dStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        ConvTranspose2dStrategy::Cube
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
pub fn conv_transpose2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvTransposeOptions<2>,
    strategy: ConvTranspose2dStrategy,
) -> JitTensor<R> {
    match strategy {
        ConvTranspose2dStrategy::Direct => {
            let client = &input.client;

            let out = empty_device::<R, E>(
                client.clone(),
                input.device.clone(),
                shape_out(&input, &weight, &options),
            );

            cubecl::convolution::conv_transpose2d::conv_transpose_direct::launch_ref::<R, E>(
                client,
                input.as_handle_ref(),
                weight.as_handle_ref(),
                bias.as_ref().map(|bias_ref| bias_ref.as_handle_ref()),
                out.as_handle_ref(),
                options.into(),
            );

            out
        }
        ConvTranspose2dStrategy::Gemm => {
            col2im::conv_transpose2d_col2im::<R, E>(input, weight, bias, options)
        }
        #[cfg(feature = "autotune")]
        ConvTranspose2dStrategy::Autotune => {
            conv_transpose2d_autotune::<R, E>(input, weight, bias, options)
        }
    }
}

pub(crate) fn shape_out<R: JitRuntime>(
    input: &JitTensor<R>,
    weight: &JitTensor<R>,
    options: &ConvTransposeOptions<2>,
) -> Shape {
    let [batch_size, _, in_height, in_width] = input.shape.dims[0..4]
        .try_into()
        .expect("Input shape should have 4 dimensions");
    let [_, out_channels, kernel_0, kernel_1] = weight.shape.dims[0..4]
        .try_into()
        .expect("Weight shape should have 4 dimensions");

    let out_0 = calculate_conv_transpose_output_size(
        kernel_0,
        options.stride[0],
        options.padding[0],
        options.padding_out[0],
        options.dilation[0],
        in_height,
    );
    let out_1 = calculate_conv_transpose_output_size(
        kernel_1,
        options.stride[1],
        options.padding[1],
        options.padding_out[1],
        options.dilation[1],
        in_width,
    );

    Shape::new([batch_size, out_channels * options.groups, out_0, out_1])
}
