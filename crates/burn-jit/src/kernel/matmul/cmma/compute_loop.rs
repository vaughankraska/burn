use burn_cube::prelude::*;

use super::base::SharedMemories;
use super::config::CmmaConfig;

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float>(
    shared_memories: SharedMemories<F>,
    config: Comptime<CmmaConfig>,
) {
}
