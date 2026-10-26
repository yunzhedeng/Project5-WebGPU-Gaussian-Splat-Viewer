const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos: vec2<f32>,        // NDC center (kept f32 for sub-pixel precision)
    size: u32,             // pack2x16float(quad half-size in NDC)
    color_rg: u32,         // pack2x16float(r, g)
    color_b_opacity: u32,  // pack2x16float(b, opacity)
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
};  // 32 bytes

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_buffer: array<u32>;   // f16 SH coefs, see load.ts

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    // load.ts layout per gaussian: 16 coefs x 3 channels (RGB) of f16
    //   = 48 f16 = 24 u32.  Coef c starts at f16 index c*3 inside that block.
    // Two f16 share one u32, so an even coef starts at the low half of a u32
    // and an odd coef starts at the high half.
    let base = splat_idx * 24u + (c_idx * 3u) / 2u;   // u32 index
    let p0 = unpack2x16float(sh_buffer[base]);
    let p1 = unpack2x16float(sh_buffer[base + 1u]);
    if ((c_idx & 1u) == 0u) {
        return vec3<f32>(p0.x, p0.y, p1.x);   // [r g][b .]
    }
    return vec3<f32>(p0.y, p1.x, p1.y);       // [. r][g b]
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) { return; }

    // 1. unpack position (stored as f16 pairs)
    let g = gaussians[idx];
    let a = unpack2x16float(g.pos_opacity[0]);   // (x, y)
    let b = unpack2x16float(g.pos_opacity[1]);   // (z, opacity)
    let pos_world = vec4<f32>(a.x, a.y, b.x, 1.0);

    // 2. world -> view -> clip -> NDC
    let pos_view = camera.view * pos_world;      // camera looks down +z
    let pos_clip = camera.proj * pos_view;
    let ndc = pos_clip.xy / pos_clip.w;

    // 3. view frustum culling (1.2x screen box keeps edge gaussians)
    if (pos_view.z < 0.2) { return; }
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) { return; }

    // 4. 3D covariance  Sigma = R S S^T R^T   (same as 3DGS computeCov3D)
    let r01 = unpack2x16float(g.rot[0]);
    let r23 = unpack2x16float(g.rot[1]);
    let q = normalize(vec4<f32>(r01.x, r01.y, r23.x, r23.y));   // (w, x, y, z)
    let s01 = unpack2x16float(g.scale[0]);
    let s23 = unpack2x16float(g.scale[1]);
    // scale is stored in log space -> exp; multiplied by the UI slider
    let scale = exp(vec3<f32>(s01.x, s01.y, s23.x)) * render_settings.gaussian_scaling;

    let r = q.x; let x = q.y; let y = q.z; let z = q.w;
    // WGSL matrix constructors are column-major, same as glm in the CUDA code
    let R = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z),       2.0 * (x * z + r * y),
        2.0 * (x * y + r * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y),       2.0 * (y * z + r * x),       1.0 - 2.0 * (x * x + y * y)
    );
    let S = mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    let M = S * R;
    let Sigma = transpose(M) * M;

    // 5. 2D covariance  cov2D = J W Sigma W^T J^T   (same as 3DGS computeCov2D)
    let t = pos_view.xyz;
    let focal = camera.focal;
    let tan_fov = camera.viewport / (2.0 * focal);
    let lim = 1.3 * tan_fov;
    let tx = clamp(t.x / t.z, -lim.x, lim.x) * t.z;
    let ty = clamp(t.y / t.z, -lim.y, lim.y) * t.z;

    // Jacobian of the perspective projection (pixel units)
    let J = mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * tx) / (t.z * t.z),
        0.0, focal.y / t.z, -(focal.y * ty) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    // rotation part of the view matrix (transposed, to match the CUDA layout)
    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    let cov = transpose(T) * Sigma * T;

    // low-pass filter: every splat at least ~1 pixel wide
    let cov2d = vec3<f32>(cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3);

    // 6. conic (= inverse of cov2D) and radius (3 sigma of the major axis)
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 0.0) { return; }
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius_px = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // pixels -> NDC (NDC spans 2 units across the viewport)
    let radius_ndc = 2.0 * radius_px / camera.viewport;

    // quad completely off screen -> cull
    if (abs(ndc.x) - radius_ndc.x > 1.0 || abs(ndc.y) - radius_ndc.y > 1.0) { return; }

    let opacity = 1.0 / (1.0 + exp(-b.y));   // sigmoid

    // view-dependent color from spherical harmonics
    let cam_pos = camera.view_inv[3].xyz;            // camera position in world space
    let dir = normalize(pos_world.xyz - cam_pos);    // camera -> gaussian
    let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));

    // 7. reserve a compact slot for this visible gaussian
    let out_idx = atomicAdd(&sort_infos.keys_size, 1u);

    // 8. write 2D splat
    var s: Splat;
    s.pos = ndc;
    s.size = pack2x16float(radius_ndc);
    s.color_rg = pack2x16float(color.rg);
    s.color_b_opacity = pack2x16float(vec2<f32>(color.b, opacity));
    s.conic_x = conic.x;
    s.conic_y = conic.y;
    s.conic_z = conic.z;
    splats[out_idx] = s;

    // 9. sort key + payload. Radix sort is ascending, we want FAR first
    //    (back-to-front), so invert the depth bits. bitcast of a positive
    //    f32 keeps ordering.
    sort_depths[out_idx] = 0xFFFFFFFFu - bitcast<u32>(pos_view.z);
    sort_indices[out_idx] = out_idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (out_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}