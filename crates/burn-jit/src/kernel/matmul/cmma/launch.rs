use std::cmp::max;

use burn_cube::{frontend::TensorArg, Compiler};

use crate::{
    kernel::{
        into_contiguous,
        matmul::cmma::{
            base::cmma_kernel,
            config::{cmma_cube_count, cmma_cube_dim, CmmaConfig},
        },
    },
    tensor::{JitTensor, MatrixLayout},
    FloatElement, JitRuntime,
};

// Only those values supported at the moment
const BLOCK_SIZE_M: usize = 64;
const BLOCK_SIZE_K: usize = 32;
const BLOCK_SIZE_N: usize = 64;

/// Matrix multiplication using tiling 2d algorithm
pub fn matmul_cmma<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    assert!(
        BLOCK_SIZE_K * max(BLOCK_SIZE_M, BLOCK_SIZE_N)
            <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );
    assert!(
        BLOCK_SIZE_M * BLOCK_SIZE_N <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );

    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let check_layout = |tensor: JitTensor<R, E, D>| match tensor.matrix_layout() {
        MatrixLayout::Contiguous => (tensor, false),
        MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (tensor, transposed),
        MatrixLayout::HighlyPermuted => (into_contiguous(tensor), false),
    };
    let (lhs, lhs_transposed) = check_layout(lhs);
    let (rhs, rhs_transposed) = check_layout(rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let lhs_vectorization = match lhs_transposed {
        true => panic!(),
        false => vectorization(k),
    };
    let rhs_vectorization = match rhs_transposed {
        true => 1,
        false => vectorization(n),
    };
    let out_vectorization = vectorization(n);

    let cube_count = cmma_cube_count::<R, D>(&out.shape, 64, 64);
    let cube_dim = cmma_cube_dim();
    let cube_config = CmmaConfig::new(m, k, n, lhs_transposed, rhs_transposed);

    assert!(lhs_vectorization == 4 && rhs_vectorization == 4 && out_vectorization == 4);

    // cmma_kernel::launch::<E::FloatPrimitive, <half::f16 as FloatElement>::FloatPrimitive, R>(
    cmma_kernel::launch::<E::FloatPrimitive, E::FloatPrimitive, R>( // TMP for WGPU testing
        client,
        cube_count,
        cube_dim,
        TensorArg::vectorized(
            lhs_vectorization,
            &lhs.handle,
            &lhs.strides,
            &lhs.shape.dims,
        ),
        TensorArg::vectorized(
            rhs_vectorization,
            &rhs.handle,
            &rhs.strides,
            &rhs.shape.dims,
        ),
        TensorArg::vectorized(
            out_vectorization,
            &out.handle,
            &out.strides,
            &out.shape.dims,
        ),
        cube_config,
    );

    out
}
