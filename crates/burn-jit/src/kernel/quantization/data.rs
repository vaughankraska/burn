use cubecl::prelude::*;

/// Number of quantized values packed into a single [QTensor](super::QTensor) element.
pub const NUM_PACKED_QINT8: u32 = 4;

/// Symmetric quantization max range mapping value for int8.
pub const SYMMETRIC_RANGE_MAX_I8: f32 = i8::MAX as f32;
/// Symmetric quantization min range mapping value for int8.
pub const SYMMETRIC_RANGE_MIN_I8: f32 = -SYMMETRIC_RANGE_MAX_I8;
/// Affine quantization max range mapping value for int8.
pub const AFFINE_RANGE_MAX_I8: f32 = i8::MAX as f32;
/// Affine quantization min range mapping value for int8.
pub const AFFINE_RANGE_MIN_I8: f32 = i8::MIN as f32;

#[cube]
/// Pack a line of 4 signed 8-bit integer values into a single unsigned 32-bit integer.
pub(crate) fn pack_i8s_into_u32(value: Line<u32>) -> u32 {
    let mut v_packed = 0;

    #[unroll]
    for i in 0..value.size() {
        // Shift and combine into u32
        v_packed |= (value[i] & 0xFF) << (8 * i);
    }
    v_packed
}

#[cube]
/// Unpack/extract the signed 8-bit integer value previously packed into the u32 value at
/// the specified offset.
pub(crate) fn unpack_i8(value: u32, offset: u32) -> i32 {
    // Extract 8-bit segment
    let value = (value >> offset) & 0xFF;
    // Check if the value is negative by inspecting the MSB and subtract 256 if it is
    // Subtract 0 or 256 to circumvent unsupported conditional assignment (let x = if {} else {};)
    let sub = i32::cast_from(value & 0x80 != 0) * 256;
    i32::cast_from(value) - sub
}

#[cube]
/// Unpack/extract all signed 8-bit integer values previously packed into a single 32-bit integer.
pub(crate) fn unpack_i8s(value: u32) -> Line<i32> {
    let mut line = Line::empty(NUM_PACKED_QINT8);
    // Extract each 8-bit segment
    #[unroll]
    for i in 0..comptime!(NUM_PACKED_QINT8) {
        line[i] = unpack_i8(value, 8 * i)
    }

    line
}
