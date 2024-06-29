@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<vec4<f32>>;

@group(0)
@binding(1)
var<storage, read_write> input_1_global: array<vec4<f32>>;

@group(0)
@binding(2)
var<storage, read_write> output_0_global: array<vec4<f32>>;

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
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var a_0_0: array<f32, 16>;

    var a_0_1: array<vec4<f32>, 4>;

    var a_0_2: array<vec4<f32>, 4>;

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
    var l_0_14: vec4<f32>;
    var l_0_15: vec4<f32>;
    var l_0_16: f32;
    var l_0_17: f32;
    l_0_0 = 64u - 1u;
    l_0_0 = l_0_0 / 4u;
    l_0_0 = l_0_0 + 1u;
    l_0_1 = workgroup_id.x * 64u;
    l_0_2 = workgroup_id.y * 64u;
    l_0_3 = local_idx / l_0_0;
    l_0_3 = l_0_3 * 4u;
    l_0_0 = local_idx % l_0_0;
    l_0_0 = l_0_0 * 4u;
    l_0_4 = u32(rank);
    l_0_5 = l_0_4 - 2u;
    l_0_6 = info[(0u * rank_2) + rank + l_0_5 + 1u];
    l_0_5 = l_0_4 - 1u;
    l_0_7 = info[(1u * rank_2) + rank + l_0_5 + 1u];
    l_0_6 = l_0_6 * l_0_7;
    l_0_6 = l_0_6 * workgroup_id.z;
    l_0_7 = u32(0u);
    l_0_5 = u32(0u);
    l_0_4 = l_0_4 - 2u;

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_4; l_1_0++) {
        l_0_8 = info[(2u * rank_2) + l_1_0 + 1u];
        l_0_8 = l_0_6 / l_0_8;
        l_0_9 = info[(0u * rank_2) + rank + l_1_0 + 1u];
        l_0_9 = l_0_8 % l_0_9;
        l_0_10 = info[(0u * rank_2) + l_1_0 + 1u];
        l_0_9 = l_0_9 * l_0_10;
        l_0_7 = l_0_7 + l_0_9;
        l_0_10 = info[(1u * rank_2) + rank + l_1_0 + 1u];
        l_0_8 = l_0_8 % l_0_10;
        l_0_10 = info[(1u * rank_2) + l_1_0 + 1u];
        l_0_8 = l_0_8 * l_0_10;
        l_0_5 = l_0_5 + l_0_8;
    }

    for (var l_1_0: u32 = 0u; l_1_0 < 16u; l_1_0++) {
        a_0_0[l_1_0] = f32(0f);
    }
    l_0_10 = rank - 1u;
    l_0_9 = info[(0u * rank_2) + rank + l_0_10 + 1u];
    l_0_10 = u32(0u);
    l_0_8 = l_0_9 / 32u;
    l_0_10 = u32(l_0_8);

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_10; l_1_0++) {
        l_0_9 = l_1_0 * 32u;
        l_0_8 = l_0_0 * 64u;
        l_0_8 = l_0_8 + l_0_3;
        l_0_4 = rank - 2u;
        l_0_11 = info[(0u * rank_2) + l_0_4 + 1u];
        l_0_11 = l_0_1 * l_0_11;
        l_0_11 = l_0_11 + l_0_9;
        l_0_11 = l_0_11 + l_0_7;
        l_0_4 = u32(rank);
        l_0_4 = l_0_4 - 2u;
        l_0_12 = info[(0u * rank_2) + l_0_4 + 1u];
        l_0_4 = l_0_3 * l_0_12;
        l_0_4 = l_0_4 + l_0_0;
        l_0_4 = l_0_4 + l_0_11;

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_0_11 = l_2_0 * l_0_12;
            l_0_11 = l_0_4 + l_0_11;
            l_0_13 = l_0_11 / 4u;
            l_0_14 = vec4<f32>(input_0_global[l_0_13]);
            a_0_1[l_2_0] = vec4<f32>(l_0_14);
        }

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_0_14[0u] = f32(0f);
            l_0_14[1u] = f32(0f);
            l_0_14[2u] = f32(0f);
            l_0_14[3u] = f32(0f);
            l_0_15 = vec4<f32>(a_0_1[0u]);
            l_0_16 = f32(l_0_15[l_2_0]);
            l_0_14[0u] = f32(l_0_16);
            l_0_15 = vec4<f32>(a_0_1[1u]);
            l_0_16 = f32(l_0_15[l_2_0]);
            l_0_14[1u] = f32(l_0_16);
            l_0_15 = vec4<f32>(a_0_1[2u]);
            l_0_16 = f32(l_0_15[l_2_0]);
            l_0_14[2u] = f32(l_0_16);
            l_0_15 = vec4<f32>(a_0_1[3u]);
            l_0_16 = f32(l_0_15[l_2_0]);
            l_0_14[3u] = f32(l_0_16);
            l_0_13 = l_2_0 * 64u;
            l_0_13 = l_0_8 + l_0_13;
            l_0_13 = l_0_13 / 4u;
            shared_memory_0[l_0_13] = vec4<f32>(l_0_14);
        }
        l_0_13 = rank - 2u;
        l_0_12 = info[(1u * rank_2) + l_0_13 + 1u];
        l_0_13 = l_0_3 * 64u;
        l_0_13 = l_0_13 + l_0_0;
        l_0_12 = l_0_9 * l_0_12;
        l_0_12 = l_0_2 + l_0_12;
        l_0_12 = l_0_12 + l_0_5;
        l_0_11 = u32(rank);
        l_0_11 = l_0_11 - 2u;
        l_0_8 = info[(1u * rank_2) + l_0_11 + 1u];
        l_0_11 = l_0_3 * l_0_8;
        l_0_11 = l_0_11 + l_0_0;
        l_0_11 = l_0_11 + l_0_12;

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_0_12 = l_2_0 * l_0_8;
            l_0_12 = l_0_11 + l_0_12;
            l_0_4 = l_0_12 / 4u;
            l_0_15 = vec4<f32>(input_1_global[l_0_4]);
            a_0_2[l_2_0] = vec4<f32>(l_0_15);
        }

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            l_0_12 = l_2_0 * 64u;
            l_0_12 = l_0_13 + l_0_12;
            l_0_12 = l_0_12 / 4u;
            l_0_15 = vec4<f32>(a_0_2[l_2_0]);
            shared_memory_1[l_0_12] = vec4<f32>(l_0_15);
        }
        workgroupBarrier();
        l_0_13 = u32(l_0_3);
        l_0_12 = u32(l_0_0);

        for (var l_2_0: u32 = 0u; l_2_0 < 32u; l_2_0++) {
            l_0_11 = l_2_0 * 64u;
            l_0_11 = l_0_13 + l_0_11;
            l_0_11 = l_0_11 / 4u;
            l_0_15 = vec4<f32>(shared_memory_0[l_0_11]);
            l_0_11 = l_2_0 * 64u;
            l_0_11 = l_0_12 + l_0_11;
            l_0_11 = l_0_11 / 4u;
            l_0_14 = vec4<f32>(shared_memory_1[l_0_11]);

            for (var l_3_0: u32 = 0u; l_3_0 < 4u; l_3_0++) {
                l_0_11 = l_3_0 * 4u;

                for (var l_4_0: u32 = 0u; l_4_0 < 4u; l_4_0++) {
                    l_0_16 = f32(l_0_15[l_3_0]);
                    l_0_17 = f32(l_0_14[l_4_0]);
                    l_0_16 = l_0_16 * l_0_17;
                    l_0_9 = l_0_11 + l_4_0;
                    l_0_17 = f32(a_0_0[l_0_9]);
                    l_0_17 = l_0_17 + l_0_16;
                    a_0_0[l_0_9] = f32(l_0_17);
                }
            }
        }
        workgroupBarrier();
    }
    l_0_13 = l_0_1 + l_0_3;
    l_0_0 = l_0_2 + l_0_0;
    l_0_12 = rank - 2u;
    l_0_11 = info[(2u * rank_2) + l_0_12 + 1u];

    for (var l_1_0: u32 = 0u; l_1_0 < 4u; l_1_0++) {
        l_0_12 = l_1_0 * 4u;
        l_0_10 = l_0_13 + l_1_0;
        l_0_10 = l_0_10 * l_0_11;
        l_0_10 = l_0_10 + l_0_0;
        l_0_10 = l_0_10 + l_0_6;

        for (var l_2_0: u32 = 0u; l_2_0 < 1u; l_2_0++) {
            l_0_15[0u] = f32(0f);
            l_0_15[1u] = f32(0f);
            l_0_15[2u] = f32(0f);
            l_0_15[3u] = f32(0f);

            for (var l_3_0: u32 = 0u; l_3_0 < 4u; l_3_0++) {
                l_0_9 = l_2_0 * 4u;
                l_0_9 = l_0_9 + l_3_0;
                l_0_9 = l_0_12 + l_0_9;
                l_0_17 = f32(a_0_0[l_0_9]);
                l_0_15[l_3_0] = f32(l_0_17);
            }
            l_0_9 = l_0_10 / 4u;
            l_0_9 = l_2_0 + l_0_9;
            output_0_global[l_0_9] = vec4<f32>(l_0_15);
        }
    }
}