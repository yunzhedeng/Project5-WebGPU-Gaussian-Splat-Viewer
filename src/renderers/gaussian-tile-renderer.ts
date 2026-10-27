import { PointCloud } from '../utils/load';
import tilePreprocessWGSL from '../shaders/tile_preprocess.wgsl';
import tileRasterWGSL from '../shaders/tile_raster.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { GaussianRenderer, dispatch_size } from './gaussian-renderer';
import { GpuTimer } from '../utils/gpu-timer';

const TILE_SIZE = 16;
const MAX_TILES = 1 << 16;          // tile_id must fit in 16 bits of the sort key
const KEYS_PER_GAUSSIAN = 16;       // key buffer = num_points * this (average tiles per splat)
const MAX_KEYS = 16_000_000;        // keeps identify_ranges dispatch under 65535 workgroups

const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  canvas: { width: number, height: number },
): GaussianRenderer {

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================
  const keys_per_sort_wg = C.histogram_wg_size * c_histogram_block_rows;
  const capacity = Math.min(
    pc.num_points * KEYS_PER_GAUSSIAN,
    MAX_KEYS,
    Math.floor(device.limits.maxStorageBufferBindingSize / 4) - 2 * keys_per_sort_wg,
  );
  console.log(`tile renderer key capacity: ${capacity}`);
  const sorter = get_sorter(capacity, device);
  const timer = new GpuTimer(device);

  const zero = new Uint32Array([0]);

  const splat_buffer = createBuffer(device, 'tile splats', pc.num_points * 32, GPUBufferUsage.STORAGE);
  const splat_count_buffer = createBuffer(device, 'tile splat count', 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const ranges_buffer = createBuffer(device, 'tile ranges', MAX_TILES * 8,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const range_dispatch_buffer = createBuffer(device, 'tile range dispatch', 3 * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, new Uint32Array([0, 1, 1]));

  // struct RenderSettings { gaussian_scaling, sh_deg, frustum_culling } (+ padding)
  const render_settings_data = new Float32Array([1.0, pc.sh_deg, 1.0, 0]);
  const render_settings_buffer = createBuffer(device, 'tile render settings', 4 * 4,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, render_settings_data);

  // ===============================================
  //    Compute Pipelines and Bind Groups
  // ===============================================
  const preprocess_module = device.createShaderModule({ code: tilePreprocessWGSL });

  const raster_module = device.createShaderModule({ code: tileRasterWGSL });
  const raster_constants = { capacity, keysPerSortWorkgroup: keys_per_sort_wg };
  const finalize_pipeline = device.createComputePipeline({
    label: 'tile finalize', layout: 'auto',
    compute: { module: raster_module, entryPoint: 'finalize', constants: raster_constants },
  });
  const ranges_pipeline = device.createComputePipeline({
    label: 'tile ranges', layout: 'auto',
    compute: { module: raster_module, entryPoint: 'identify_ranges', constants: raster_constants },
  });
  const rasterize_pipeline = device.createComputePipeline({
    label: 'tile rasterize', layout: 'auto',
    compute: { module: raster_module, entryPoint: 'rasterize', constants: raster_constants },
  });
  const blit_pipeline = device.createRenderPipeline({
    label: 'tile blit', layout: 'auto',
    vertex: { module: raster_module, entryPoint: 'vs_blit', constants: raster_constants },
    fragment: { module: raster_module, entryPoint: 'fs_blit', constants: raster_constants,
      targets: [{ format: presentation_format }] },
    primitive: { topology: 'triangle-list' },
  });

  // after 4 radix passes the sorted keys/values are back in ping_pong[0]
  const keys_buffer = sorter.ping_pong[0].sort_depths_buffer;
  const values_buffer = sorter.ping_pong[0].sort_indices_buffer;

  // preprocess pipeline depends on the workgroup size (UI) -> rebuildable
  let workgroup_size = C.histogram_wg_size;
  let preprocess_pipeline: GPUComputePipeline;
  let pre_bind_groups: GPUBindGroup[];
  const build_preprocess = (wg_size: number) => {
    workgroup_size = wg_size;
    preprocess_pipeline = device.createComputePipeline({
      label: 'tile preprocess',
      layout: 'auto',
      compute: { module: preprocess_module, entryPoint: 'preprocess', constants: { workgroupSize: wg_size } },
    });
    pre_bind_groups = [
      device.createBindGroup({ label: 'tile pre camera', layout: preprocess_pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: camera_buffer } }] }),
      device.createBindGroup({ label: 'tile pre gaussians', layout: preprocess_pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
          { binding: 1, resource: { buffer: splat_buffer } },
          { binding: 2, resource: { buffer: pc.sh_buffer } },
        ] }),
      device.createBindGroup({ label: 'tile pre keys', layout: preprocess_pipeline.getBindGroupLayout(2),
        entries: [
          { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
          { binding: 1, resource: { buffer: keys_buffer } },
          { binding: 2, resource: { buffer: values_buffer } },
          { binding: 3, resource: { buffer: splat_count_buffer } },
        ] }),
      device.createBindGroup({ label: 'tile pre settings', layout: preprocess_pipeline.getBindGroupLayout(3),
        entries: [{ binding: 0, resource: { buffer: render_settings_buffer } }] }),
    ];
  };
  build_preprocess(C.histogram_wg_size);

  const finalize_bind_group = device.createBindGroup({
    label: 'tile finalize', layout: finalize_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      { binding: 2, resource: { buffer: range_dispatch_buffer } },
    ],
  });

  const ranges_bind_group = device.createBindGroup({
    label: 'tile ranges', layout: ranges_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: keys_buffer } },
      { binding: 2, resource: { buffer: ranges_buffer } },
    ],
  });

  // output texture depends on the canvas size -> (re)created on resize
  let out_width = 0, out_height = 0;
  let out_texture: GPUTexture | undefined;
  let raster_bind_group: GPUBindGroup;
  let blit_bind_group: GPUBindGroup;

  const ensure_output = () => {
    const w = Math.max(1, canvas.width), h = Math.max(1, canvas.height);
    if (out_texture && w === out_width && h === out_height) return;
    if (Math.ceil(w / TILE_SIZE) * Math.ceil(h / TILE_SIZE) > MAX_TILES) {
      console.warn('canvas too large for 16-bit tile ids');
    }
    out_texture?.destroy();
    out_width = w; out_height = h;
    out_texture = device.createTexture({
      label: 'tile output', size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = out_texture.createView();
    raster_bind_group = device.createBindGroup({
      label: 'tile rasterize', layout: rasterize_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: splat_buffer } },
        { binding: 2, resource: { buffer: values_buffer } },
        { binding: 3, resource: { buffer: ranges_buffer } },
        { binding: 4, resource: view },
      ],
    });
    blit_bind_group = device.createBindGroup({
      label: 'tile blit', layout: blit_pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: view }],
    });
  };

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const compute_pass = (encoder: GPUCommandEncoder, label: string,
                        pipeline: GPUComputePipeline, groups: GPUBindGroup[],
                        dispatch: (p: GPUComputePassEncoder) => void,
                        timestampWrites?: GPUComputePassTimestampWrites) => {
    const pass = encoder.beginComputePass({ label, timestampWrites });
    pass.setPipeline(pipeline);
    groups.forEach((g, i) => pass.setBindGroup(i, g));
    dispatch(pass);
    pass.end();
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      ensure_output();
      const tiles_x = Math.ceil(out_width / TILE_SIZE);
      const tiles_y = Math.ceil(out_height / TILE_SIZE);

      // reset counters / ranges
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, zero);   // keys_size = 0
      device.queue.writeBuffer(splat_count_buffer, 0, zero);
      encoder.clearBuffer(ranges_buffer, 0, tiles_x * tiles_y * 8);

      compute_pass(encoder, 'tile preprocess', preprocess_pipeline, pre_bind_groups,
        p => p.dispatchWorkgroups(...dispatch_size(pc.num_points, workgroup_size)), timer.writes(0, 1));
      compute_pass(encoder, 'tile finalize', finalize_pipeline, [finalize_bind_group],
        p => p.dispatchWorkgroups(1));
      sorter.sort(encoder);
      // timestamp 2 = start of the render phase (ranges + rasterize + blit); sort = t2 - t1
      compute_pass(encoder, 'tile ranges', ranges_pipeline, [ranges_bind_group],
        p => p.dispatchWorkgroupsIndirect(range_dispatch_buffer, 0), timer.writes(2, undefined));
      compute_pass(encoder, 'tile rasterize', rasterize_pipeline, [raster_bind_group],
        p => p.dispatchWorkgroups(tiles_x, tiles_y));

      const pass = encoder.beginRenderPass({
        label: 'tile blit',
        colorAttachments: [{ view: texture_view, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }],
        timestampWrites: timer.writes(undefined, 3),
      });
      pass.setPipeline(blit_pipeline);
      pass.setBindGroup(0, blit_bind_group);
      pass.draw(3);
      pass.end();
      timer.resolve(encoder, splat_count_buffer);   // count = visible splats
    },
    camera_buffer,
    timer,
    set_gaussian_multiplier: (value: number) => {
      render_settings_data[0] = value;
      device.queue.writeBuffer(render_settings_buffer, 0, render_settings_data);
    },
    set_frustum_culling: (on: boolean) => {
      render_settings_data[2] = on ? 1.0 : 0.0;
      device.queue.writeBuffer(render_settings_buffer, 0, render_settings_data);
    },
    set_workgroup_size: (size: number) => build_preprocess(size),
  };
}