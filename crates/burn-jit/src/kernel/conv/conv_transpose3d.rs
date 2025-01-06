use crate::{ops::numeric::empty_device, tensor::JitTensor, FloatElement, JitRuntime};
use burn_tensor::{ops::ConvTransposeOptions, Shape};

pub(crate) fn conv_transpose3d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvTransposeOptions<3>,
) -> JitTensor<R> {
    let client = &input.client;

    let out = empty_device::<R, E>(
        client.clone(),
        input.device.clone(),
        shape_out(&input, &weight, &options),
    );

    cubecl::convolution::conv_transpose3d::launch_ref::<R, E>(
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
    options: &ConvTransposeOptions<3>,
) -> Shape {
    let [batch_size, _, in_depth, in_height, in_width] = input.shape.dims();
    let [_, out_channels, kernel_0, kernel_1, kernel_2] = weight.shape.dims();

    let out_0 = (in_depth - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_height - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;
    let out_2 = (in_width - 1) * options.stride[2]
        + options.dilation[2] * (kernel_2 - 1)
        + options.padding_out[2]
        - 2 * options.padding[2]
        + 1;

    Shape::new([
        batch_size,
        out_channels * options.groups,
        out_0,
        out_1,
        out_2,
    ])
}
