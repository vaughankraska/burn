@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> input_1_global: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> info: array<u32>;

var<workgroup> shared_memory_0: array<vec4<f32>, 512>;

var<workgroup> shared_memory_1: array<vec4<f32>, 512>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {

    var a_0_0: array<f32, 16>;

    let rank: u32 = info[0];
    let rank_2: u32 = rank * 2u;
    var l_0_0: u32;
    var l_0_1: u32;
    var l_0_2: u32;
    var l_0_3: u32;
    var l_0_4: u32;
    var l_0_5: u32;
    var l_0_6: u32;
    var l_0_7: u32;
    var l_0_8: u32;
    var l_0_9: u32;
    var l_0_10: u32;
    var l_0_11: u32;
    var l_0_12: u32;
    var l_0_13: u32;
    var l_0_14: u32;
    var l_0_15: u32;
    var l_0_16: u32;
    var l_0_17: u32;
    var l_0_18: u32;
    var l_0_19: u32;
    var l_0_20: u32;
    var l_0_21: u32;
    var l_0_22: u32;
    var l_0_23: u32;
    var l_0_24: u32;
    var l_0_25: u32;
    var l_0_26: u32;
    var l_0_27: u32;
    var l_0_28: u32;
    var l_0_29: vec4<f32>;
    var l_0_30: vec4<f32>;
    var l_0_31: u32;
    var l_0_32: u32;
    var l_0_33: f32;
    var l_0_34: f32;
    var l_0_35: f32;
    var l_0_36: u32;
    var l_0_37: u32;
    var l_0_38: bool;
    var l_0_39: bool;
    l_0_0 = rank - 1u;
    l_0_1 = rank - 2u;
    l_0_2 = info[(0u * rank_2) + rank + l_0_1 + 1u];
    l_0_3 = info[(0u * rank_2) + rank + l_0_0 + 1u];
    l_0_4 = info[(1u * rank_2) + rank + l_0_0 + 1u];
    l_0_5 = info[(0u * rank_2) + l_0_1 + 1u];
    l_0_6 = info[(0u * rank_2) + l_0_0 + 1u];
    l_0_7 = info[(1u * rank_2) + l_0_1 + 1u];
    l_0_8 = info[(1u * rank_2) + l_0_0 + 1u];
    l_0_9 = info[(2u * rank_2) + l_0_1 + 1u];
    l_0_10 = info[(2u * rank_2) + l_0_0 + 1u];
    l_0_11 = u32(workgroup_id.x);
    l_0_11 = l_0_11 * 64u;
    l_0_12 = u32(workgroup_id.y);
    l_0_12 = l_0_12 * 64u;
    l_0_13 = local_idx / 16u;
    l_0_13 = l_0_13 * 4u;
    l_0_14 = local_idx % 16u;
    l_0_14 = l_0_14 * 4u;
    l_0_15 = l_0_11 + l_0_13;
    l_0_16 = l_0_12 + l_0_14;
    l_0_17 = l_0_11 * l_0_5;
    l_0_18 = l_0_12 * l_0_8;
    l_0_19 = l_0_2 * l_0_4;
    l_0_19 = l_0_19 * global_id.z;
    l_0_20 = rank - 2u;

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_20; l_1_0++) {
        l_0_21 = info[(0u * rank_2) + l_1_0 + 1u];
        l_0_22 = info[(1u * rank_2) + l_1_0 + 1u];
        l_0_23 = info[(2u * rank_2) + l_1_0 + 1u];
        l_0_24 = info[(0u * rank_2) + rank + l_1_0 + 1u];
        l_0_25 = info[(1u * rank_2) + rank + l_1_0 + 1u];
        l_0_26 = l_0_19 / l_0_23;
        l_0_27 = l_0_26 % l_0_24;
        l_0_27 = l_0_27 * l_0_21;
        l_0_17 = l_0_17 + l_0_27;
        l_0_28 = l_0_26 % l_0_25;
        l_0_28 = l_0_28 * l_0_22;
        l_0_18 = l_0_18 + l_0_28;
    }
    l_0_33 = f32(l_0_3);
    l_0_34 = f32(32u);
    l_0_35 = l_0_33 / l_0_34;
    l_0_35 = ceil(l_0_35);
    l_0_31 = u32(l_0_35);

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_31; l_1_0++) {
        var l_1_1: u32;
        var l_1_2: u32;
        var l_1_3: bool;
        var l_1_4: u32;
        var l_1_5: bool;
        var l_1_6: u32;
        var l_1_7: bool;
        var l_1_8: bool;
        var l_1_9: vec4<f32>;
        var l_1_10: u32;
        var l_1_11: u32;
        var l_1_12: u32;
        var l_1_13: u32;
        var l_1_14: u32;
        var l_1_15: bool;
        var l_1_16: f32;
        var l_1_17: f32;
        var l_1_18: f32;
        var l_1_19: f32;
        var l_1_20: u32;
        var l_1_21: u32;
        var l_1_22: bool;
        var l_1_23: u32;
        var l_1_24: bool;
        var l_1_25: u32;
        var l_1_26: bool;
        var l_1_27: bool;
        var l_1_28: vec4<f32>;
        var l_1_29: u32;
        var l_1_30: u32;
        var l_1_31: u32;
        var l_1_32: u32;
        var l_1_33: u32;
        var l_1_34: bool;
        var l_1_35: f32;
        var l_1_36: f32;
        var l_1_37: f32;
        var l_1_38: f32;
        var l_1_39: u32;
        var l_1_40: u32;
        var l_1_41: f32;
        var l_1_42: f32;
        var l_1_43: f32;
        var l_1_44: u32;
        var l_1_45: f32;
        var l_1_46: f32;
        l_0_32 = l_1_0 * 32u;
        l_1_1 = l_0_2 - l_0_15;

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_1_2 = l_0_14 + l_2_0;
            l_1_3 = l_1_2 < 32u;
            if l_1_3 {
                l_1_4 = l_0_13 / 4u;
                l_1_4 = l_1_4 * 32u;
                l_1_4 = l_1_4 + l_1_2;
                l_1_6 = l_1_2 + l_0_32;
                l_1_5 = l_1_6 < l_0_3;
                l_1_7 = l_1_1 >= 1u;
                l_1_8 = l_1_5 && l_1_7;
                if l_1_8 {
                    var l_4_0: u32;
                    l_1_11 = l_0_32 + l_1_2;
                    l_1_11 = l_1_11 * l_0_6;
                    l_1_10 = l_0_13 * l_0_5;
                    l_1_11 = l_1_11 + l_1_10;
                    l_1_11 = l_1_11 + l_0_17;
                    l_1_12 = l_1_11 + l_0_5;
                    l_1_13 = l_1_12 + l_0_5;
                    l_1_14 = l_1_13 + l_0_5;
                    l_1_15 = l_1_1 >= 4u;
                    if l_1_15 {
                        l_1_16 = f32(input_0_global[l_1_11]);
                        l_1_17 = f32(input_0_global[l_1_12]);
                        l_1_18 = f32(input_0_global[l_1_13]);
                        l_1_19 = f32(input_0_global[l_1_14]);
                    } else {
                        l_1_15 = l_1_1 == 3u;
                        if l_1_15 {
                            l_1_16 = f32(input_0_global[l_1_11]);
                            l_1_17 = f32(input_0_global[l_1_12]);
                            l_1_18 = f32(input_0_global[l_1_13]);
                            l_1_19 = f32(0u);
                        } else {
                            l_1_15 = l_1_1 == 2u;
                            if l_1_15 {
                                l_1_16 = f32(input_0_global[l_1_11]);
                                l_1_17 = f32(input_0_global[l_1_12]);
                                l_1_18 = f32(0u);
                                l_1_19 = f32(0u);
                            } else {
                                l_1_15 = l_1_1 == 1u;
                                if l_1_15 {
                                    l_1_16 = f32(input_0_global[l_1_11]);
                                    l_1_17 = f32(0u);
                                    l_1_18 = f32(0u);
                                    l_1_19 = f32(0u);
                                }
                            }
                        }
                    }
                    l_4_0 = u32(0u);
                    l_1_9[l_4_0] = f32(l_1_16);
                    l_4_0 = l_4_0 + 1u;
                    l_1_9[l_4_0] = f32(l_1_17);
                    l_4_0 = l_4_0 + 1u;
                    l_1_9[l_4_0] = f32(l_1_18);
                    l_4_0 = l_4_0 + 1u;
                    l_1_9[l_4_0] = f32(l_1_19);
                    shared_memory_0[l_1_4] = vec4<f32>(l_1_9);
                } else {
                    var l_4_0: u32;
                    l_1_16 = f32(0u);
                    l_4_0 = u32(0u);
                    l_1_9[l_4_0] = f32(l_1_16);
                    l_4_0 = l_4_0 + 1u;
                    l_1_9[l_4_0] = f32(l_1_16);
                    l_4_0 = l_4_0 + 1u;
                    l_1_9[l_4_0] = f32(l_1_16);
                    l_4_0 = l_4_0 + 1u;
                    l_1_9[l_4_0] = f32(l_1_16);
                    shared_memory_0[l_1_4] = vec4<f32>(l_1_9);
                }
            }
        }
        l_1_20 = l_0_4 - l_0_16;

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_1_21 = l_0_13 + l_2_0;
            l_1_22 = l_1_21 < 32u;
            if l_1_22 {
                l_1_23 = l_1_21 * 64u;
                l_1_23 = l_1_23 + l_0_14;
                l_1_23 = l_1_23 / 4u;
                l_1_25 = l_1_21 + l_0_32;
                l_1_24 = l_1_25 < l_0_3;
                l_1_26 = l_1_20 >= 1u;
                l_1_27 = l_1_24 && l_1_26;
                if l_1_27 {
                    var l_4_0: u32;
                    l_1_30 = l_0_32 + l_1_21;
                    l_1_30 = l_1_30 * l_0_7;
                    l_1_29 = l_0_14 * l_0_8;
                    l_1_30 = l_1_30 + l_1_29;
                    l_1_30 = l_1_30 + l_0_18;
                    l_1_31 = l_1_30 + l_0_8;
                    l_1_32 = l_1_31 + l_0_8;
                    l_1_33 = l_1_32 + l_0_8;
                    l_1_34 = l_1_20 >= 4u;
                    if l_1_34 {
                        l_1_35 = f32(input_1_global[l_1_30]);
                        l_1_36 = f32(input_1_global[l_1_31]);
                        l_1_37 = f32(input_1_global[l_1_32]);
                        l_1_38 = f32(input_1_global[l_1_33]);
                    } else {
                        l_1_34 = l_1_20 == 3u;
                        if l_1_34 {
                            l_1_35 = f32(input_1_global[l_1_30]);
                            l_1_36 = f32(input_1_global[l_1_31]);
                            l_1_37 = f32(input_1_global[l_1_32]);
                            l_1_38 = f32(0u);
                        } else {
                            l_1_34 = l_1_20 == 2u;
                            if l_1_34 {
                                l_1_35 = f32(input_1_global[l_1_30]);
                                l_1_36 = f32(input_1_global[l_1_31]);
                                l_1_37 = f32(0u);
                                l_1_38 = f32(0u);
                            } else {
                                l_1_34 = l_1_20 == 1u;
                                if l_1_34 {
                                    l_1_35 = f32(input_1_global[l_1_30]);
                                    l_1_36 = f32(0u);
                                    l_1_37 = f32(0u);
                                    l_1_38 = f32(0u);
                                }
                            }
                        }
                    }
                    l_4_0 = u32(0u);
                    l_1_28[l_4_0] = f32(l_1_35);
                    l_4_0 = l_4_0 + 1u;
                    l_1_28[l_4_0] = f32(l_1_36);
                    l_4_0 = l_4_0 + 1u;
                    l_1_28[l_4_0] = f32(l_1_37);
                    l_4_0 = l_4_0 + 1u;
                    l_1_28[l_4_0] = f32(l_1_38);
                    shared_memory_1[l_1_23] = vec4<f32>(l_1_28);
                } else {
                    var l_4_0: u32;
                    l_1_35 = f32(0u);
                    l_4_0 = u32(0u);
                    l_1_28[l_4_0] = f32(l_1_35);
                    l_4_0 = l_4_0 + 1u;
                    l_1_28[l_4_0] = f32(l_1_35);
                    l_4_0 = l_4_0 + 1u;
                    l_1_28[l_4_0] = f32(l_1_35);
                    l_4_0 = l_4_0 + 1u;
                    l_1_28[l_4_0] = f32(l_1_35);
                    shared_memory_1[l_1_23] = vec4<f32>(l_1_28);
                }
            }
        }
        workgroupBarrier();

        for (var l_2_0: u32 = 0u; l_2_0 < 32u; l_2_0++) {
            l_1_39 = l_0_13 / 4u;
            l_1_39 = l_1_39 * 32u;
            l_1_39 = l_1_39 + l_2_0;
            l_0_29 = vec4<f32>(shared_memory_0[l_1_39]);
            l_1_40 = l_2_0 * 64u;
            l_1_40 = l_1_40 + l_0_14;
            l_1_40 = l_1_40 / 4u;
            l_0_30 = vec4<f32>(shared_memory_1[l_1_40]);

            for (var l_3_0: u32 = 0u; l_3_0 < 4u; l_3_0++) {

                for (var l_4_0: u32 = 0u; l_4_0 < 4u; l_4_0++) {
                    l_1_41 = f32(l_0_29[l_3_0]);
                    l_1_42 = f32(l_0_30[l_4_0]);
                    l_1_43 = l_1_41 * l_1_42;
                    l_1_44 = l_3_0 * 4u;
                    l_1_44 = l_1_44 + l_4_0;
                    l_1_45 = f32(a_0_0[l_1_44]);
                    l_1_46 = l_1_45 + l_1_43;
                    a_0_0[l_1_44] = f32(l_1_46);
                }
            }
        }
        workgroupBarrier();
    }

    for (var l_1_0: u32 = 0u; l_1_0 < 4u; l_1_0++) {

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_0_36 = l_0_15 + l_1_0;
            l_0_37 = l_0_16 + l_2_0;
            l_0_38 = l_0_36 < l_0_2;
            l_0_39 = l_0_37 < l_0_4;
            l_0_38 = l_0_38 && l_0_39;
            if l_0_38 {
                var l_3_0: u32;
                var l_3_1: f32;
                var l_3_2: u32;
                l_3_0 = l_1_0 * 4u;
                l_3_0 = l_3_0 + l_2_0;
                l_3_1 = f32(a_0_0[l_3_0]);
                l_0_36 = l_0_36 * l_0_9;
                l_0_37 = l_0_37 * l_0_10;
                l_3_2 = l_0_36 + l_0_37;
                l_3_2 = l_3_2 + l_0_19;
                output_0_global[l_3_2] = f32(l_3_1);
            }
        }
    }
}