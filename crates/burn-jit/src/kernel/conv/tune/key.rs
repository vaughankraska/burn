use burn_tensor::DType;
use cubecl::{
    convolution::{conv2d::ConvolutionProblem, ConvOptions},
    linalg::matmul::components::MatrixLayout,
    tensor_line_size, AutotuneKey,
};
use serde::{Deserialize, Serialize};

use crate::{FloatElement, JitRuntime};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct Conv2dAutotuneKey {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    #[autotune(anchor)]
    pub height: usize,
    #[autotune(anchor)]
    pub width: usize,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
    pub dtype: DType,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct ConvTranspose2dAutotuneKey {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    #[autotune(anchor)]
    pub height: usize,
    #[autotune(anchor)]
    pub width: usize,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
    pub dtype: DType,
}

pub fn problem_from_key<R: JitRuntime, F: FloatElement>(
    key: &Conv2dAutotuneKey,
    out_h: usize,
    out_w: usize,
) -> ConvolutionProblem {
    let in_stride_2 = key.in_channels;
    let in_stride_1 = key.width * in_stride_2;
    let in_stride_0 = key.height * in_stride_1;

    let m = key.batch_size * out_h * out_w;
    let n = key.out_channels;
    let k = key.kernel_size[0] * key.kernel_size[1] * key.in_channels;

    let options = ConvOptions {
        stride: key.stride,
        padding: key.padding,
        dilation: key.dilation,
        groups: key.groups,
    };

    // Target 128 bit accesses
    let available_vectorizations = R::supported_line_sizes()
        .iter()
        .copied()
        .filter(|it| *it as usize * size_of::<F>() <= 16)
        .collect::<Vec<_>>();
    let lhs_line_size = tensor_line_size(
        &available_vectorizations,
        &[key.batch_size, key.height, key.width, key.in_channels],
        &[in_stride_0, in_stride_1, in_stride_2, 1],
        3,
    );
    let rhs_line_size = tensor_line_size(&available_vectorizations, &[k, n], &[n, 1], 1);
    let out_line_size = tensor_line_size(&available_vectorizations, &[m, n], &[n, 1], 1);

    ConvolutionProblem {
        m,
        n,
        k,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        lhs_line_size,
        rhs_line_size,
        out_line_size,
        kernel_size: (key.kernel_size[0] as u32, key.kernel_size[1] as u32),
        options,
        out_shape_y: out_h,
        out_shape_x: out_w,
        has_bias: key.has_bias,
    }
}
