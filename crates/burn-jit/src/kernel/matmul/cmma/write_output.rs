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
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_k);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let sm_vec = Comptime::map(config, |c| c.sm_vec);

    let num_tiles_per_subcube = Comptime::runtime(block_size_k / tile_size); // 2
    let tile_size_r = Comptime::runtime(tile_size); // 16
    let sm_vec_r = Comptime::runtime(sm_vec); // 4
    let accumulate_sm_stride = Comptime::runtime(block_size_n / sm_vec); // 16

    let out_stride = dims.n;

    let within_tile_row_offset = SUBCUBE_DIM / sm_vec_r; // assuming subcube_dim is 32 -> 8
    let within_sm_row_offset = SUBCUBE_DIM / Comptime::runtime(tile_size); // assuming subcube_dim is 32 -> 2
    let subcube_id = UNIT_POS_X;
    let id_within_subcube = UNIT_POS_Y;

    // There are two because 32 / 16. TODO generalize
    let unit_read_row_0 = id_within_subcube / accumulate_sm_stride;
    let unit_read_row_1 = unit_read_row_0 + within_sm_row_offset;
    let unit_read_col = id_within_subcube % accumulate_sm_stride;

    let unit_write_row_0 = id_within_subcube / sm_vec_r;
    let unit_write_row_1 = unit_write_row_0 + within_tile_row_offset;
    let unit_write_col = id_within_subcube % sm_vec_r;

    for n_iter in range(0u32, 2u32, Comptime::new(true)) {
        let num_row_offset = Comptime::runtime(tile_size * tile_size / block_size_n); // 4
        let row_offset = (subcube_id + n_iter) * num_row_offset;

        let read_pos_0 = (row_offset + unit_read_row_0) * accumulate_sm_stride + unit_read_col;
        let read_pos_1 = (row_offset + unit_read_row_1) * accumulate_sm_stride + unit_read_col;

        let tile_row = subcube_id / num_tiles_per_subcube;
        let tile_col = (subcube_id % num_tiles_per_subcube) * num_tiles_per_subcube + n_iter;

        let out_offset = offsets.batch_out + offsets.cube_row * out_stride + offsets.cube_col;
        let out_write_tile_offset =
            out_offset + tile_row * tile_size_r * out_stride + tile_col * tile_size_r;

        let out_write_pos_0 =
            out_write_tile_offset + unit_write_row_0 * out_stride + unit_write_col;
        let out_write_pos_1 =
            out_write_tile_offset + unit_write_row_1 * out_stride + unit_write_col;

        out[out_write_pos_0] = accumulate[read_pos_0];
        out[out_write_pos_1] = accumulate[read_pos_1];
    }
}
