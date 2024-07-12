use burn_cube::prelude::*;

use super::base::SharedMemories;
use super::config::CmmaConfig;

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float, FC: Float>(
    mut shared_memories: SharedMemories<F, FC>,
    config: Comptime<CmmaConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let sm_vec = Comptime::map(config, |c| c.sm_vec);
    let num_tiles_in_k = Comptime::runtime(block_size_k / tile_size); // 2
    let num_tile_elems = Comptime::runtime((tile_size * tile_size) / sm_vec); // 256 / 4

    let num_tiles_per_row = block_size_m / tile_size; // 4
    let num_tiles_per_col = block_size_n / tile_size; // 4
    let num_tiles = num_tiles_per_row * num_tiles_per_col; // 16

    let n_iterations = Comptime::runtime(num_tiles) / CUBE_DIM_X; // 2
    let num_subcube_per_row =
        Comptime::runtime(block_size_n) / (n_iterations * Comptime::runtime(tile_size)); // 2

    let subcube_id = UNIT_POS_X;
    let tile_row = subcube_id / num_subcube_per_row;
    let tile_col_base = (subcube_id % num_subcube_per_row) * n_iterations;

    for n_iter in range(0u32, n_iterations, Comptime::new(false)) {
        let tile_col = tile_col_base + n_iter;
        let accumulate_pos =
            (tile_row * Comptime::runtime(num_tiles_per_row) + tile_col) * num_tile_elems;
        let accumulate_slice = shared_memories
            .accumulate
            .slice_mut(accumulate_pos, accumulate_pos + num_tile_elems);

        for k_iter in range(0u32, num_tiles_in_k, Comptime::new(false)) {
            let shared_lhs_pos = (num_tiles_in_k * tile_row + k_iter) * num_tile_elems;
            let shared_rhs_pos = (num_tiles_in_k * tile_col + k_iter) * num_tile_elems;

            let lhs_slice = shared_memories
                .lhs
                .slice(shared_lhs_pos, shared_lhs_pos + num_tile_elems);
            let rhs_slice = shared_memories
                .rhs
                .slice(shared_rhs_pos, shared_rhs_pos + num_tile_elems);

            cmma_computation(lhs_slice, rhs_slice, accumulate_slice);
        }
    }
}

#[cube]
pub fn cmma_computation<F: Float, FC: Float>(
    lhs: &Slice<FC>,
    rhs: &Slice<FC>,
    out: &mut SliceMut<F>,
) {
    let a = cmma::Matrix::<FC>::new(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    let b = cmma::Matrix::<FC>::new(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
    );
    let c = cmma::Matrix::<F>::new(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
    );
    cmma::fill::<F>(&c, F::new(0.0));
    cmma::load::<FC>(&a, lhs.as_slice(), UInt::new(16));
    cmma::load::<FC>(&b, rhs.as_slice(), UInt::new(16));

    cmma::execute::<FC, FC, F, F>(&a, &b, &c, &c);

    cmma::store::<F>(
        out.as_slice_mut(),
        &c,
        UInt::new(16),
        cmma::MatrixLayout::RowMajor,
    );
}
