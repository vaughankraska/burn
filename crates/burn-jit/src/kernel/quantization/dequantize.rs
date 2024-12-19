use crate::tensor::JitTensor;
use crate::FloatElement;
use crate::{JitElement, JitRuntime};
use burn_tensor::quantization::{QuantizationScheme, QuantizationType};
use burn_tensor::DType;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

use super::{unpack_i8s, QParams, QTensor};

#[cube]
/// Recover the floating-point value for an integer value obtained via int8 symmetric quantization.
pub(crate) fn dequantize_symmetric_int8<F: Float>(value: Line<i32>, scale: f32) -> Line<F> {
    // x = scale * x_q
    Line::cast_from(scale) * Line::cast_from(value)
}

#[cube]
/// Recover the floating-point value for an integer value obtained via int8 affine quantization.
pub(crate) fn dequantize_affine_int8<F: Float>(
    value: Line<i32>,
    scale: f32,
    offset: i32,
) -> Line<F> {
    // x = scale * (x_q - offset)
    Line::cast_from(scale) * Line::cast_from(value - Line::cast_from(offset))
}

#[cube]
/// Dequantize the packed 8-bit signed integer values to recover the 4 original floating-point values.
pub(crate) fn dequantize_symmetric_int8_unpacked<F: Float>(value: u32, scale: f32) -> Line<F> {
    dequantize_symmetric_int8(unpack_i8s(value), scale)
}

#[cube]
/// Dequantize the packed 8-bit signed integer values to recover the 4 original floating-point values.
pub(crate) fn dequantize_affine_int8_unpacked<F: Float>(
    value: u32,
    scale: f32,
    offset: i32,
) -> Line<F> {
    dequantize_affine_int8(unpack_i8s(value), scale, offset)
}

#[cube]
/// Dequantize multiple packed 8-bit signed integer values to recover the original floating-point values.
///
/// # Note
/// This function assumes that the input line size is a factor of 4.
pub(crate) fn dequantize_symmetric_int8_unpacked_line<F: Float>(
    value: Line<u32>,
    scale: f32,
) -> Line<F> {
    let line_size = value.size();
    let num_packed = crate::kernel::quantization::NUM_PACKED_QINT8;
    let mut values = Line::<F>::empty(line_size * num_packed);
    #[unroll]
    for i in 0..line_size {
        let v = dequantize_symmetric_int8_unpacked(value[i], scale);
        #[unroll]
        for j in 0..num_packed {
            values[i * num_packed + j] = v[j];
        }
    }
    values
}

#[cube]
/// Dequantize multiple packed 8-bit signed integer values to recover the original floating-point values.
///
/// # Note
/// This function assumes that the input line size is a factor of 4.
pub(crate) fn dequantize_affine_int8_unpacked_line<F: Float>(
    value: Line<u32>,
    scale: f32,
    offset: i32,
) -> Line<F> {
    let line_size = value.size();
    let num_packed = crate::kernel::quantization::NUM_PACKED_QINT8;
    let mut values = Line::<F>::empty(line_size * num_packed);
    #[unroll]
    for i in 0..line_size {
        let v = dequantize_affine_int8_unpacked(value[i], scale, offset);
        #[unroll]
        for j in 0..num_packed {
            values[i * num_packed + j] = v[j];
        }
    }
    values
}

#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_affine_int8_kernel(
    input: &QTensor,
    output: &mut Tensor<Line<f32>>,
    #[comptime] scheme: QuantizationScheme,
) {
    // Last two positions contain the qparams
    if ABSOLUTE_POS >= input.len() - 2 {
        return;
    }

    let qparams = QParams::new(scheme);
    let (scale, offset) = qparams.values(input);

    let value = input[ABSOLUTE_POS];

    // Input line size is fixed to 1
    if comptime!(output.line_size() == 4) {
        output[ABSOLUTE_POS] = dequantize_affine_int8_unpacked::<f32>(value[0], scale, offset);
    } else {
        // For very small inputs where number of elements < 4, the output line size is 1
        let out = dequantize_affine_int8_unpacked::<f32>(value[0], scale, offset);

        #[unroll]
        for j in 0..out.size() {
            output[ABSOLUTE_POS + j] = Line::cast_from(out[j]);
        }
    }
}

// Would have wrapped symmetric with the same affine kernel but cube doesn't support Option<Tensor> for offset.
#[cube(launch_unchecked)]
pub(crate) fn dequantize_per_tensor_symmetric_int8_kernel(
    input: &QTensor,
    output: &mut Tensor<Line<f32>>,
    #[comptime] scheme: QuantizationScheme,
) {
    // Last position contains the qparam
    if ABSOLUTE_POS >= input.len() - 1 {
        return;
    }

    let qparams = QParams::new(scheme);
    let (scale, _) = qparams.values(input);

    let value = input[ABSOLUTE_POS];

    // Input line size is fixed to 1
    if comptime!(output.line_size() == 4) {
        output[ABSOLUTE_POS] = dequantize_symmetric_int8_unpacked::<f32>(value[0], scale);
    } else {
        // For very small inputs where number of elements < 4, the output line size is 1
        let out = dequantize_symmetric_int8_unpacked::<f32>(value[0], scale);

        #[unroll]
        for j in 0..out.size() {
            output[ABSOLUTE_POS + j] = Line::cast_from(out[j]);
        }
    }
}

pub(crate) fn dequantize_per_tensor<R, F>(tensor: JitTensor<R>) -> JitTensor<R>
where
    R: JitRuntime,
    F: JitElement,
{
    // The actual number of elements is 1/4 (four int8 values packed in a single u32)
    // so we choose a line size to match a valid input binding size.
    let num_out_elems = tensor.shape.num_elements();
    let num_elems = usize::div_ceil(num_out_elems, 4);
    let line_size_in = 1;
    let line_size_out = if num_out_elems < 4 { 1 } else { 4 };
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size_in as usize, cube_dim);

    let client = tensor.client.clone();
    let handle = client.empty(num_out_elems * core::mem::size_of::<F>());

    let output = JitTensor::new_contiguous(
        client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        handle,
        F::dtype(),
    );

    if let DType::QFloat(scheme) = tensor.dtype {
        match scheme {
            QuantizationScheme::PerTensorAffine(QuantizationType::QInt8) => {
                unsafe {
                    dequantize_per_tensor_affine_int8_kernel::launch_unchecked::<R>(
                        &client,
                        cube_count,
                        cube_dim,
                        tensor.as_array_arg::<u32>(line_size_in),
                        output.as_tensor_arg::<F>(line_size_out),
                        scheme,
                    )
                };
            }
            QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8) => {
                unsafe {
                    dequantize_per_tensor_symmetric_int8_kernel::launch_unchecked::<R>(
                        &client,
                        cube_count,
                        cube_dim,
                        tensor.as_array_arg::<u32>(line_size_in),
                        output.as_tensor_arg::<F>(line_size_out),
                        scheme,
                    )
                };
            }
        }
    }

    output
}

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F>(tensor: JitTensor<R>) -> JitTensor<R>
where
    R: JitRuntime,
    F: FloatElement,
{
    dequantize_per_tensor::<R, F>(tensor)
}
