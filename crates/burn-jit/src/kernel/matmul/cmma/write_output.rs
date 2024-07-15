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
    let out_vec = Comptime::vectorization(out);
    let out_vec_r = Comptime::runtime(out_vec);

    let num_tiles_per_subcube = Comptime::runtime(block_size_k / tile_size); // 2
    let acc_sm_stride = Comptime::runtime(block_size_n); // 64
    let acc_sm_stride_vec = Comptime::runtime(block_size_n / out_vec); // Even if not really vectorized, because write out_vec_r values

    let out_stride = dims.n;

    let subcube_dim = UInt::new(32);
    let within_tile_row_offset = subcube_dim / out_vec_r; // assuming subcube_dim is 32 -> 8
    let within_sm_row_offset = subcube_dim * out_vec_r / acc_sm_stride; // assuming subcube_dim is 32 -> 2
    let subcube_id = UNIT_POS_X;
    let id_within_subcube = UNIT_POS_Y;

    // There are two because 32 / 16. TODO generalize
    let unit_read_row_0 = id_within_subcube / acc_sm_stride_vec;
    let unit_read_row_1 = unit_read_row_0 + within_sm_row_offset;
    let unit_read_col = id_within_subcube % acc_sm_stride_vec;

    let n_per_row = Comptime::runtime(tile_size / out_vec); // 4
    let unit_write_row_0 = id_within_subcube / n_per_row;
    let unit_write_row_1 = unit_write_row_0 + within_tile_row_offset;
    let unit_write_col = id_within_subcube % n_per_row;

    for n_iter in range(0u32, num_tiles_per_subcube, Comptime::new(true)) {
        let single_row_offset = Comptime::runtime(tile_size * tile_size / block_size_n); // 4
        let row_offset = (subcube_id + n_iter) * single_row_offset;

        let read_pos_0 = (row_offset + unit_read_row_0) * acc_sm_stride + unit_read_col * out_vec_r;
        let read_pos_1 = (row_offset + unit_read_row_1) * acc_sm_stride + unit_read_col * out_vec_r;

        let tile_row = subcube_id / num_tiles_per_subcube;
        let tile_col = (subcube_id % num_tiles_per_subcube) * num_tiles_per_subcube + n_iter;

        let total_row_0 = tile_row + unit_write_row_0;
        let total_row_1 = tile_row + unit_write_row_1;
        let total_col = tile_col * n_per_row + unit_write_col;

        let out_offset = offsets.batch_out + offsets.cube_row * out_stride + offsets.cube_col;

        let out_write_pos_0 = out_offset + total_row_0 * out_stride + total_col * out_vec_r;
        let out_write_pos_1 = out_offset + total_row_1 * out_stride + total_col * out_vec_r;

        let mut a = F::vectorized_empty(Comptime::get(out_vec));
        for i in range(0u32, 4u32, Comptime::new(true)) {
            a[i] = accumulate[read_pos_0 + i];
        }
        out[out_write_pos_0 / out_vec_r] = a;

        let mut b = F::vectorized_empty(Comptime::get(out_vec));
        for i in range(0u32, 4u32, Comptime::new(true)) {
            b[i] = accumulate[read_pos_1 + i];
        }
        out[out_write_pos_1 / out_vec_r] = b;
    }
}

#[cfg(feature = "export_tests")]
/// Compute loop exported tests
pub mod tests {

    use crate::kernel::matmul::cmma::base::{DimensionsExpand, OffsetsExpand};
    use crate::kernel::matmul::test_utils::{assert_equals, zeros_tensor};
    use crate::{kernel::matmul::test_utils::range_tensor, JitRuntime};

    use super::*;

    #[cube(launch)]
    fn write_output_test<F: Float>(
        out: &mut Tensor<F>,
        acc_sm_arr: &mut Array<F>, // TODO can't have a non-mut array?
        config: Comptime<CmmaConfig>,
    ) {
        let offsets = Offsets {
            batch_lhs: UInt::new(0),
            batch_rhs: UInt::new(0),
            batch_out: UInt::new(0),
            cube_row: UInt::new(0),
            cube_col: UInt::new(0),
            k: UInt::new(0),
        };

        let mut accumulate = SharedMemory::<F>::new(4096);
        for i in range(0u32, 4096u32, Comptime::new(false)) {
            accumulate[i] = acc_sm_arr[i];
        }

        let dims = Dimensions {
            m: UInt::new(16),
            k: UInt::new(32),
            n: UInt::new(32),
        };

        write_to_output(out, accumulate, offsets, dims, config);
    }

    /// Exported test
    pub fn cmma_write_output_unit_test<R: JitRuntime>(device: &R::Device) {
        let out = zeros_tensor::<R>(16, 32, device);
        let acc_sm = range_tensor::<R>(64, 64, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count: CubeCount<R::Server> = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        write_output_test::launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(4, &out.handle, &out.strides, &out.shape.dims),
            ArrayArg::new(&acc_sm.handle, 64 * 64),
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 256.0,
            257.0, 258.0, 259.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 128.0, 129.0, 130.0, 131.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 384.0, 385.0, 386.0, 387.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(out.handle, expected, device);
    }

    /// Exported test
    pub fn cmma_write_output_warp_test<R: JitRuntime>(device: &R::Device) {
        let out = zeros_tensor::<R>(16, 32, device);
        let acc_sm = range_tensor::<R>(64, 64, device);
        let cube_dim = CubeDim::new(1, 32, 1);
        let cube_count: CubeCount<R::Server> = CubeCount::Static(1, 1, 1);

        let config = CmmaConfig {
            block_size_m: UInt::new(64),
            block_size_k: UInt::new(32),
            block_size_n: UInt::new(64),
            check_m_bounds: false,
            check_k_bounds: false,
            check_n_bounds: false,
            tile_size: UInt::new(16),
            sm_vec: UInt::new(4),
            lhs_transposed: false,
            rhs_transposed: false,
            unroll: false,
        };

        write_output_test::launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(4, &out.handle, &out.strides, &out.shape.dims),
            ArrayArg::new(&acc_sm.handle, 64 * 64),
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            256.0, 257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0, 267.0,
            268.0, 269.0, 270.0, 271.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 272.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0,
            279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 32.0, 33.0, 34.0, 35.0,
            36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 288.0, 289.0,
            290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0,
            302.0, 303.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
            60.0, 61.0, 62.0, 63.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0,
            313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 319.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
            70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 320.0, 321.0, 322.0, 323.0,
            324.0, 325.0, 326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0,
            80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0,
            94.0, 95.0, 336.0, 337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 344.0, 345.0,
            346.0, 347.0, 348.0, 349.0, 350.0, 351.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0,
            103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 352.0, 353.0, 354.0,
            355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0, 365.0, 366.0,
            367.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
            123.0, 124.0, 125.0, 126.0, 127.0, 368.0, 369.0, 370.0, 371.0, 372.0, 373.0, 374.0,
            375.0, 376.0, 377.0, 378.0, 379.0, 380.0, 381.0, 382.0, 383.0, 128.0, 129.0, 130.0,
            131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0,
            143.0, 384.0, 385.0, 386.0, 387.0, 388.0, 389.0, 390.0, 391.0, 392.0, 393.0, 394.0,
            395.0, 396.0, 397.0, 398.0, 399.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0,
            151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 400.0, 401.0, 402.0,
            403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 410.0, 411.0, 412.0, 413.0, 414.0,
            415.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0,
            171.0, 172.0, 173.0, 174.0, 175.0, 416.0, 417.0, 418.0, 419.0, 420.0, 421.0, 422.0,
            423.0, 424.0, 425.0, 426.0, 427.0, 428.0, 429.0, 430.0, 431.0, 176.0, 177.0, 178.0,
            179.0, 180.0, 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0,
            191.0, 432.0, 433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 439.0, 440.0, 441.0, 442.0,
            443.0, 444.0, 445.0, 446.0, 447.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0,
            199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 448.0, 449.0, 450.0,
            451.0, 452.0, 453.0, 454.0, 455.0, 456.0, 457.0, 458.0, 459.0, 460.0, 461.0, 462.0,
            463.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0,
            219.0, 220.0, 221.0, 222.0, 223.0, 464.0, 465.0, 466.0, 467.0, 468.0, 469.0, 470.0,
            471.0, 472.0, 473.0, 474.0, 475.0, 476.0, 477.0, 478.0, 479.0, 224.0, 225.0, 226.0,
            227.0, 228.0, 229.0, 230.0, 231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0,
            239.0, 480.0, 481.0, 482.0, 483.0, 484.0, 485.0, 486.0, 487.0, 488.0, 489.0, 490.0,
            491.0, 492.0, 493.0, 494.0, 495.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0,
            247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 496.0, 497.0, 498.0,
            499.0, 500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 508.0, 509.0, 510.0,
            511.0,
        ];
        assert_equals::<R>(out.handle, expected, device);
    }
}
