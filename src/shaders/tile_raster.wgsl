const TILE_SIZE: u32 = 16u;
const DEPTH_BITS: u32 = 16u;
const BATCH: u32 = 256u;            // = TILE_SIZE * TILE_SIZE threads

override capacity: u32;             // max number of keys (size of sort_values)
override keysPerSortWorkgroup: u32; // radix sort: histogram_wg_size * block_rows

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    pos: vec2<f32>,
    size: u32,
    color_rg: u32,
    color_b_opacity: u32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct DispatchIndirect {
    x: u32,
    y: u32,
    z: u32,
}

// ------------------------------------------------------------------ finalize
@group(0) @binding(0) var<storage, read_write> fin_sort_infos: SortInfos;
@group(0) @binding(1) var<storage, read_write> fin_sort_dispatch: DispatchIndirect;
@group(0) @binding(2) var<storage, read_write> fin_range_dispatch: DispatchIndirect;

@compute @workgroup_size(1)
fn finalize() {
    let n = min(fin_sort_infos.keys_size, capacity);   // drop keys beyond capacity
    fin_sort_infos.keys_size = n;
    fin_sort_dispatch.x = (n + keysPerSortWorkgroup - 1u) / keysPerSortWorkgroup;
    fin_range_dispatch.x = (n + 255u) / 256u;
}

// ------------------------------------------------------------ identify_ranges
@group(0) @binding(0) var<storage, read> rng_sort_infos: SortInfos;
@group(0) @binding(1) var<storage, read> rng_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> rng_ranges: array<vec2<u32>>;

@compute @workgroup_size(256)
fn identify_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = rng_sort_infos.keys_size;
    let i = gid.x;
    if (i >= n) { return; }

    let tile = rng_keys[i] >> DEPTH_BITS;
    // first key of this tile -> start, last key of this tile -> end
    if (i == 0u || (rng_keys[i - 1u] >> DEPTH_BITS) != tile) {
        rng_ranges[tile].x = i;
    }
    if (i == n - 1u || (rng_keys[i + 1u] >> DEPTH_BITS) != tile) {
        rng_ranges[tile].y = i + 1u;
    }
}

// ------------------------------------------------------------------ rasterize
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba8unorm, write>;

// one batch of splats shared by the whole tile
var<workgroup> s_xy: array<vec2<f32>, BATCH>;          // center in pixels
var<workgroup> s_conic_opacity: array<vec4<f32>, BATCH>;
var<workgroup> s_color: array<vec3<f32>, BATCH>;
var<workgroup> s_rounds: u32;
var<workgroup> s_done_count: atomic<u32>;
var<workgroup> s_all_done: bool;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let vp = camera.viewport;
    let tiles_x = (u32(vp.x) + TILE_SIZE - 1u) / TILE_SIZE;
    let tile_id = wid.y * tiles_x + wid.x;

    let pix = wid.xy * TILE_SIZE + lid.xy;
    let inside = pix.x < u32(vp.x) && pix.y < u32(vp.y);
    let pixf = vec2<f32>(pix) + 0.5;      // pixel center, same as @builtin(position)

    let range = ranges[tile_id];
    if (lidx == 0u) {
        s_rounds = (range.y - range.x + BATCH - 1u) / BATCH;
        atomicStore(&s_done_count, 0u);
    }
    let rounds = workgroupUniformLoad(&s_rounds);   // uniform loop bound (barrier inside)

    var T = 1.0;                  // transmittance
    var C = vec3<f32>(0.0);       // accumulated color
    var done = !inside;
    if (done) { atomicAdd(&s_done_count, 1u); }

    for (var r = 0u; r < rounds; r++) {
        // whole tile saturated -> stop early (3DGS __syncthreads_count)
        workgroupBarrier();
        if (lidx == 0u) { s_all_done = atomicLoad(&s_done_count) == BATCH; }
        if (workgroupUniformLoad(&s_all_done)) { break; }

        // cooperative fetch: each thread loads one splat of this batch
        let batch_start = range.x + r * BATCH;
        let idx = batch_start + lidx;
        if (idx < range.y) {
            let sp = splats[sorted_values[idx]];
            s_xy[lidx] = vec2<f32>((sp.pos.x * 0.5 + 0.5) * vp.x, (0.5 - sp.pos.y * 0.5) * vp.y);
            let b_o = unpack2x16float(sp.color_b_opacity);
            s_conic_opacity[lidx] = vec4<f32>(sp.conic_x, sp.conic_y, sp.conic_z, b_o.y);
            s_color[lidx] = vec3<f32>(unpack2x16float(sp.color_rg), b_o.x);
        }
        workgroupBarrier();

        // front-to-back blending (same as 3DGS renderCUDA)
        let count = min(BATCH, range.y - batch_start);
        for (var j = 0u; j < count && !done; j++) {
            let d = s_xy[j] - pixf;
            let co = s_conic_opacity[j];
            let power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0.0) { continue; }

            let alpha = min(0.99, co.w * exp(power));
            if (alpha < 1.0 / 255.0) { continue; }

            let test_T = T * (1.0 - alpha);
            if (test_T < 0.0001) {
                done = true;
                atomicAdd(&s_done_count, 1u);
                break;
            }
            C += s_color[j] * alpha * T;
            T = test_T;
        }
    }

    if (inside) {
        // background is black, so final = C + T * 0
        textureStore(out_tex, vec2<i32>(pix), vec4<f32>(C, 1.0));
    }
}

// ----------------------------------------------------------------------- blit
@group(0) @binding(0) var blit_src: texture_2d<f32>;

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // one triangle covering the whole screen
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4<f32>(p, 0.0, 1.0);
}

@fragment
fn fs_blit(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(blit_src, vec2<i32>(p.xy), 0);
}