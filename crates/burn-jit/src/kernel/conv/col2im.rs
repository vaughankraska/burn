use burn_tensor::{
    ops::{conv::calculate_conv_transpose_output_size, ConvTransposeOptions},
    Shape,
};
use cubecl::convolution::conv2d::batches_per_run;

use crate::{
    kernel::{
        into_contiguous,
        matmul::{matmul, MatmulStrategy},
        slice,
    },
    ops::{numeric::empty_device, reshape, swap_dims},
    tensor::JitTensor,
    FloatElement, JitElement, JitRuntime,
};

/// Perform a 2D convolution transposition using the GEMM (col2im) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_transpose2d_col2im<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R> {
    let [input_channels, im_ch_per_group, kernel_h, kernel_w] = weight.shape.dims();
    let [batch_size, _, input_h, input_w] = input.shape.dims();
    let groups = options.groups;
    let input_ch_per_group = input_channels / groups;
    let ConvTransposeOptions {
        padding: [padding_h, padding_w],
        padding_out: [padding_out_h, padding_out_w],
        dilation: [dilation_h, dilation_w],
        stride: [stride_h, stride_w],
        ..
    } = options.clone();

    let im_h = calculate_conv_transpose_output_size(
        kernel_h,
        stride_h,
        padding_h,
        padding_out_h,
        dilation_h,
        input_h,
    );
    let im_w = calculate_conv_transpose_output_size(
        kernel_w,
        stride_w,
        padding_w,
        padding_out_w,
        dilation_w,
        input_w,
    );
    let im_channels = im_ch_per_group * groups;

    let batches_per_run = batches_per_run(batch_size, input_h, input_w)
        .expect("Image too large to run even one batch at once");
    let col_shape_0 = im_ch_per_group * kernel_h * kernel_w;

    let weight = reshape(
        weight.clone(),
        Shape::new([groups, input_ch_per_group, col_shape_0]),
    );
    let weight = into_contiguous(swap_dims(weight, 1, 2));

    if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;

        let im_shape = Shape::new([runs, batches_per_run, im_channels, im_h, im_w]);
        let image = empty_device::<R, E>(input.client.clone(), input.device.clone(), im_shape);

        let input_shape = Shape::new([runs, batches_per_run, input_channels, input_h, input_w]);
        let input = reshape(input, input_shape);
        let input_shape_run = Shape::new([batches_per_run, input_channels, input_h, input_w]);

        for run in 0..runs {
            let input = col2im_index::<R, E>(input.clone(), run);
            let input = reshape(input, input_shape_run.clone());
            let im_shape = Shape::new([batches_per_run, im_channels, im_h, im_w]);
            let image_slice = col2im_index::<R, E>(image.clone(), run);
            let image_slice = reshape(image_slice, im_shape);

            execute::<R, E>(
                input,
                weight.clone(),
                bias.clone(),
                image_slice,
                options.clone(),
                kernel_h,
                kernel_w,
            );
        }

        reshape(image, Shape::new([batch_size, im_channels, im_h, im_w]))
    } else {
        let im_shape = Shape::new([batches_per_run, im_channels, im_h, im_w]);
        let image = empty_device::<R, E>(input.client.clone(), input.device.clone(), im_shape);

        execute::<R, E>(
            input,
            weight,
            bias,
            image.clone(),
            options,
            kernel_h,
            kernel_w,
        );

        image
    }
}

pub(crate) fn col2im_index<R: JitRuntime, E: JitElement>(
    tensor: JitTensor<R>,
    i: usize,
) -> JitTensor<R> {
    #[allow(clippy::single_range_in_vec_init)]
    let mut indices = vec![i..i + 1];
    for dim in tensor.shape.dims[1..].iter() {
        indices.push(0..*dim);
    }
    let new_shape = Shape {
        dims: tensor.shape.dims[1..].to_vec(),
    };
    let tensor = slice::<R, E>(tensor, &indices);
    reshape(tensor, new_shape)
}

#[allow(clippy::too_many_arguments)]
fn execute<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    image: JitTensor<R>,
    options: ConvTransposeOptions<2>,
    kernel_h: usize,
    kernel_w: usize,
) {
    let [batch_size, _, input_h, input_w] = input.shape.dims();
    let [groups, col_shape_0, input_ch_per_group] = weight.shape.dims();

    let col_shape_1 = batch_size * input_h * input_w;

    let input = swap_dims(input, 0, 1);
    let input_shape = Shape::new([groups, input_ch_per_group, col_shape_1]);
    let input = reshape(input, input_shape);

    let columns = matmul::<R, E>(weight, input, None, MatmulStrategy::default());
    let columns = reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]));

    cubecl::convolution::conv2d::col2im::<R, E>(
        &columns.client,
        columns.as_handle_ref(),
        bias.as_ref().map(|bias_ref| bias_ref.as_handle_ref()),
        image.as_handle_ref(),
        kernel_h,
        kernel_w,
        input_h,
        input_w,
        options.into(),
    );
}
