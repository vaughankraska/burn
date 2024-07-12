use burn_cube::prelude::*;

use super::{
    base::{Dimensions, Offsets, SharedMemories},
    config::CmmaConfig,
};

#[cube]
pub(crate) fn load_to_shared_memories<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    offsets: Offsets,
    shared: SharedMemories<F>,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
    load_lhs_for_cmma(lhs, shared.lhs, offsets, config, dims);
    load_rhs_for_cmma(rhs, shared.rhs, offsets, config, dims);
}

#[cube]
fn load_lhs_for_cmma<F: Float>(
    lhs: &Tensor<F>,
    shared: SharedMemory<F>,
    offsets: Offsets,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
}

#[cube]
fn load_rhs_for_cmma<F: Float>(
    lhs: &Tensor<F>,
    shared: SharedMemory<F>,
    offsets: Offsets,
    config: Comptime<CmmaConfig>,
    dims: Dimensions,
) {
}
