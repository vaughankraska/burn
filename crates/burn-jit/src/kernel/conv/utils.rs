use burn_tensor::Shape;

use crate::{kernel::slice, ops::reshape, tensor::JitTensor, JitElement, JitRuntime};

pub(crate) fn tensor_index<R: JitRuntime, E: JitElement>(
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
