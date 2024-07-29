use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use cubecl::ir::KernelDefinition;
use cubecl::linalg::tensor::index_offset_with_layout;
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};

#[cube(launch)]
fn gather_kernel<T: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<UInt>,
    output: &mut Tensor<T>,
    dim: Comptime<UInt>,
    rank: Comptime<UInt>,
) {
    let dim_runtime = Comptime::runtime(dim);
    let not_zeroth_dim = Comptime::map(dim, |d| d > 0);

    // The offset for the `dim` dimension is obtained by the indices tensor.
    let index = indices[ABSOLUTE_POS];
    let stride = input.stride(dim_runtime);

    let mut offset = index * stride;

    // We fetch the offset before the `dim` dimension.
    if Comptime::get(not_zeroth_dim) {
        offset += index_offset_with_layout(
            input,
            output,
            ABSOLUTE_POS,
            UInt::new(0),
            dim_runtime,
            Comptime::new(true),
        );
    }

    offset += index_offset_with_layout(
        input,
        output,
        ABSOLUTE_POS,
        Comptime::runtime(Comptime::map(dim, |d| d + 1)),
        Comptime::runtime(rank),
        Comptime::new(true),
    );

    output[ABSOLUTE_POS] = input[offset];
}

pub(crate) fn gather<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
) -> JitTensor<R, E, D> {
    let shape_output = indices.shape.clone();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise::<R::Server>(shape_output.num_elements(), cube_dim);

    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);

    gather_kernel::launch::<E::Primitive, R>(
        &tensor.client,
        cube_count,
        cube_dim,
        TensorArg::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        TensorArg::new(&indices.handle, &indices.strides, &indices.shape.dims),
        TensorArg::new(&output.handle, &output.strides, &output.shape.dims),
        UInt::new(dim as u32),
        UInt::new(D as u32),
    );

    output
}
