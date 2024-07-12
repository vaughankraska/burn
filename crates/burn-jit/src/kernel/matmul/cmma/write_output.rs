use burn_cube::prelude::*;

use super::{
    base::{Dimensions, Offsets},
    config::CmmaConfig,
};

#[cube]
pub(crate) fn write_to_output<F: Float>(
    out: &mut Tensor<F>,
    accumulate: SharedMemory<F>,
    offsets: Offsets,
    dims: Dimensions,
    config: Comptime<CmmaConfig>,
) {
}
