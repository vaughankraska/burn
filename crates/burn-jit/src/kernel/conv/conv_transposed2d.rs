use burn_tensor::ops::ConvTransposeOptions;

use crate::{tensor::JitTensor, FloatElement, IntElement, JitRuntime};

#[cfg(feature = "autotune")]
use super::conv_transpose2d_autotune;

/// The strategy to be used when launching a conv_transpose kernel.
pub enum ConvTranspose2dStrategy {
    /// Cube implementation
    Cube,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
}

impl Default for ConvTranspose2dStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return ConvTranspose2dStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        ConvTranspose2dStrategy::Cube
    }
}

/// Perform a 2D convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
///
pub fn conv_transpose2d<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvTransposeOptions<2>,
    strategy: ConvTranspose2dStrategy,
) -> JitTensor<R> {
    match strategy {
        ConvTranspose2dStrategy::Cube => {
            todo!()
        }
        #[cfg(feature = "autotune")]
        ConvTranspose2dStrategy::Autotune => {
            conv_transpose2d_autotune::<R, E>(input, weight, bias, options)
        }
    }
}
