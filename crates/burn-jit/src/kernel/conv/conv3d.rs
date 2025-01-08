use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

use crate::{ops::numeric::empty_device, tensor::JitTensor, FloatElement, JitRuntime};

pub(crate) fn conv3d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<3>,
) -> JitTensor<R> {
    let client = &input.client;

    let out = empty_device::<R, E>(
        client.clone(),
        input.device.clone(),
        shape_out(&input, &weight, &options),
    );

    cubecl::convolution::conv3d::launch_ref::<R, E>(
        client,
        input.as_handle_ref(),
        weight.as_handle_ref(),
        bias.as_ref().map(|bias_ref| bias_ref.as_handle_ref()),
        out.as_handle_ref(),
        options.into(),
    );

    out
}

fn shape_out<R: JitRuntime>(
    input: &JitTensor<R>,
    weight: &JitTensor<R>,
    options: &ConvOptions<3>,
) -> Shape {
    let [batch_size, _, in_depth, in_height, in_width] = input.shape.dims[0..5]
        .try_into()
        .expect("Input shape should have 5 dimensions");
    let [out_channels, _, kernel_0, kernel_1, kernel_2] = weight.shape.dims[0..5]
        .try_into()
        .expect("Weight shape should have 5 dimensions");

    let out_0 = calculate_conv_output_size(
        kernel_0,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_depth,
    );
    let out_1 = calculate_conv_output_size(
        kernel_1,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_height,
    );
    let out_2 = calculate_conv_output_size(
        kernel_2,
        options.stride[2],
        options.padding[2],
        options.dilation[2],
        in_width,
    );

    Shape::from([batch_size, out_channels, out_0, out_1, out_2])
}
