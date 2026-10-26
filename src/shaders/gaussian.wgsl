struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    // flat = same value for the whole quad, no interpolation
    @location(0) @interpolate(flat) color: vec4<f32>,      // rgb + opacity
    @location(1) @interpolate(flat) conic: vec3<f32>,      // inverse 2D covariance (pixel^-2)
    @location(2) @interpolate(flat) center_px: vec2<f32>,  // splat center in framebuffer pixels
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos: vec2<f32>,
    size: u32,
    color_rg: u32,
    color_b_opacity: u32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,      // 0..3  -> quad corner
    @builtin(instance_index) iid: u32,    // 0..keys_size-1, already depth-sorted
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    let splat = splats[sort_indices[iid]];

    // triangle-strip corners: 0:(-1,-1) 1:(1,-1) 2:(-1,1) 3:(1,1)
    let corner = vec2<f32>(f32(vid & 1u) * 2.0 - 1.0, f32(vid >> 1u) * 2.0 - 1.0);
    let size = unpack2x16float(splat.size);

    var out: VertexOutput;
    out.position = vec4<f32>(splat.pos + corner * size, 0.0, 1.0);
    out.color = vec4<f32>(unpack2x16float(splat.color_rg), unpack2x16float(splat.color_b_opacity));
    out.conic = vec3<f32>(splat.conic_x, splat.conic_y, splat.conic_z);
    // NDC -> framebuffer pixels (framebuffer y points down, NDC y points up)
    out.center_px = vec2<f32>(
        (splat.pos.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - splat.pos.y * 0.5) * camera.viewport.y
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // pixel offset between the splat center and this fragment
    let d = in.center_px - in.position.xy;

    // gaussian falloff: power = -1/2 * d^T * conic * d   (same as 3DGS forward.cu)
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    if (power > 0.0) { discard; }

    let alpha = min(0.99, in.color.a * exp(power));
    if (alpha < 1.0 / 255.0) { discard; }

    // premultiplied alpha, blended back-to-front by the pipeline blend state
    return vec4<f32>(in.color.rgb * alpha, alpha);
}