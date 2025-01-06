use burn_tensor::ops::ConvOptions;

use crate::{tensor::JitTensor, FloatElement, JitRuntime};

#[cfg(feature = "autotune")]
use super::conv2d_autotune;

/// The strategy to be used when launching a convolution kernel.
pub enum Conv2dStrategy {
    // Cube implementation
    Cube,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
}

impl Default for Conv2dStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return Conv2dStrategy::Autotune;

        // if autotune is disabled, default to the more memory-conservative algorithm
        #[cfg(not(feature = "autotune"))]
        Conv2dStrategy::Cube
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
pub fn conv2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
    strategy: Conv2dStrategy,
) -> JitTensor<R> {
    match strategy {
        Conv2dStrategy::Cube => {
            todo!()
        }
        #[cfg(feature = "autotune")]
        Conv2dStrategy::Autotune => conv2d_autotune::<R, E>(input, weight, bias, options),
    }
}
