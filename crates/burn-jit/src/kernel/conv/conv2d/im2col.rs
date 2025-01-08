use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use cubecl::convolution::conv2d::batches_per_run;

use crate::{
    kernel::{conv::utils::tensor_index, launch_binop, matmul::matmul, AddOp},
    ops::{numeric::empty_device, reshape, swap_dims},
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

/// Perform a 2D convolution using the GEMM (im2col) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_im2col<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> JitTensor<R> {
    let [batch_size, in_channels, in_height, in_width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let groups = options.groups;
    let out_c_per_group = out_channels / groups;

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

    if kernel_h == 1 && kernel_w == 1 && in_height == out_h && in_width == out_w {
        // Special case for 1x1 kernels (sometimes used to scale the image by a set of weights)
        return execute_1x1_kernel::<R, E>(input, weight, bias, options);
    }

    let batches_per_run = batches_per_run(batch_size, out_h, out_w)
        .expect("Image too large to run even one batch at once");
    let matmul_shape = Shape::new([groups, out_c_per_group, batches_per_run * out_h * out_w]);

    let mut out = if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;
        let out_shape = Shape::new([runs, out_channels, batches_per_run, out_h, out_w]);
        let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), out_shape);
        let in_shape = Shape::new([runs, batches_per_run, in_channels, in_height, in_width]);
        let input = reshape(input, in_shape);
        let in_shape_run = Shape::new([batches_per_run, in_channels, in_height, in_width]);
        for run in 0..runs {
            let input = tensor_index::<R, E>(input.clone(), run);
            let input = reshape(input, in_shape_run.clone());
            let out_slice = tensor_index::<R, E>(out.clone(), run);
            let out_slice = reshape(out_slice, matmul_shape.clone());
            execute::<R, E>(
                input,
                weight.clone(),
                out_slice,
                options.clone(),
                out_h,
                out_w,
            );
        }
        let out = swap_dims(out, 1, 2);
        reshape(out, Shape::new([batch_size, out_channels, out_h, out_w]))
    } else {
        let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), matmul_shape);
        execute::<R, E>(input, weight, out.clone(), options, out_h, out_w);
        let out = reshape(out, Shape::new([out_channels, batch_size, out_h, out_w]));
        swap_dims(out, 0, 1)
    };

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, out_channels, 1, 1]));
        out = launch_binop::<R, E, AddOp>(out, bias)
    }
    out
}

fn execute_1x1_kernel<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> JitTensor<R> {
    let [batch_size, _, height, width] = input.shape.dims();
    let [out_channels, in_c_per_grp, _, _] = weight.shape.dims();
    let groups = options.groups;
    let out_c_per_grp = out_channels / groups;

    let input = swap_dims(input, 0, 1); // [CNHW]

    let weight = reshape(weight, Shape::new([groups, out_c_per_grp, in_c_per_grp]));
    let in_shape = Shape::new([groups, in_c_per_grp, batch_size * height * width]);
    let input = reshape(input, in_shape);
    let out = matmul::<R, E>(weight, input, None, Default::default());
    let mut out = reshape(out, Shape::new([out_channels, batch_size, height, width]));

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([out_channels, 1, 1, 1]));
        out = launch_binop::<R, E, AddOp>(out, bias)
    }

    swap_dims(out, 0, 1)
}

fn execute<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    out: JitTensor<R>,
    options: ConvOptions<2>,
    out_h: usize,
    out_w: usize,
) {
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let [batch_size, in_channels, _, _] = input.shape.dims();
    let groups = options.groups;

    let col_shape_0 = in_channels * kernel_h * kernel_w;
    let col_shape_1 = batch_size * out_h * out_w;
    let shape_col = Shape::new([col_shape_0, col_shape_1]);
    let client = &input.client;
    let columns = empty_device::<R, E>(client.clone(), input.device.clone(), shape_col);

    cubecl::convolution::conv2d::im2col::<R, E>(
        client,
        input.as_handle_ref(),
        columns.as_handle_ref(),
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        options.into(),
    );

    let col_shape_0 = col_shape_0 / groups;
    let out_c_per_group = out_channels / groups;

    let columns = reshape(columns, Shape::new([groups, col_shape_0, col_shape_1]));
    let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_shape_0]));

    matmul::<R, E>(weight, columns, Some(out), Default::default());
}
