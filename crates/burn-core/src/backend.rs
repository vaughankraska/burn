#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;

#[cfg(feature = "ndarray")]
pub use ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use burn_autodiff as autodiff;

#[cfg(feature = "autodiff")]
pub use burn_autodiff::Autodiff;

#[cfg(feature = "wgpu")]
pub use burn_wgpu as wgpu;

#[cfg(feature = "wgpu")]
pub use burn_wgpu::Wgpu;

#[cfg(feature = "vulkan")]
pub use burn_wgpu as vulkan;

#[cfg(feature = "vulkan")]
pub use burn_wgpu::Vulkan;

#[cfg(feature = "cuda")]
pub use burn_cuda as cuda;

#[cfg(feature = "cuda")]
pub use burn_cuda::Cuda;

#[cfg(feature = "candle")]
pub use burn_candle as candle;

#[cfg(feature = "candle")]
pub use burn_candle::Candle;

#[cfg(feature = "libtorch")]
pub use burn_tch as libtorch;

#[cfg(feature = "libtorch")]
pub use burn_tch::LibTorch;
