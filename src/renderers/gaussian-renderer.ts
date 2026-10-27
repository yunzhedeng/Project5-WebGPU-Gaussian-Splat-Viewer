import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { GpuTimer } from '../utils/gpu-timer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
  set_frustum_culling: (on: boolean) => void,
  set_workgroup_size: (size: number) => void,
}

// Utility to create GPU buffers
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

// 1D dispatch is limited to 65535 workgroups -> spill into y for small workgroup sizes
export const dispatch_size = (n_threads: number, wg_size: number): [number, number] => {
  const groups = Math.ceil(n_threads / wg_size);
  const x = Math.min(groups, 65535);
  return [x, Math.ceil(groups / x)];
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  const timer = new GpuTimer(device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const c_size_splat = 32; // must match struct Splat in the shaders
  const splat_buffer = createBuffer(
    device, 'splats', pc.num_points * c_size_splat, GPUBufferUsage.STORAGE
  );

  // drawIndirect args: [vertexCount, instanceCount, firstVertex, firstInstance]
  const indirect_draw_buffer = createBuffer(
    device, 'indirect draw', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([4, 0, 0, 0])
  );

  // struct RenderSettings { gaussian_scaling, sh_deg, frustum_culling } (+ padding to 16 bytes)
  const render_settings_data = new Float32Array([1.0, pc.sh_deg, 1.0, 0]);
  const render_settings_buffer = createBuffer(
    device, 'render settings', 4 * 4,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings_data
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_module = device.createShaderModule({ code: preprocessWGSL });

  // the pipeline (and its 'auto' bind group layouts) depend on the workgroup
  // size, so everything is rebuilt when the UI changes it
  let workgroup_size = C.histogram_wg_size;
  let preprocess_pipeline: GPUComputePipeline;
  let preprocess_bind_groups: GPUBindGroup[];

  const build_preprocess = (wg_size: number) => {
    workgroup_size = wg_size;
    preprocess_pipeline = device.createComputePipeline({
      label: 'preprocess',
      layout: 'auto',
      compute: {
        module: preprocess_module,
        entryPoint: 'preprocess',
        constants: {
          workgroupSize: wg_size,
          sortKeyPerThread: c_histogram_block_rows,
          sortWorkgroupSize: C.histogram_wg_size,
        },
      },
    });

    const camera_bind_group = device.createBindGroup({
      label: 'preprocess camera',
      layout: preprocess_pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
    });

    const gaussian_bind_group = device.createBindGroup({
      label: 'preprocess gaussians',
      layout: preprocess_pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
        { binding: 1, resource: { buffer: splat_buffer } },
        { binding: 2, resource: { buffer: pc.sh_buffer } },
      ],
    });

    const sort_bind_group = device.createBindGroup({
      label: 'sort',
      layout: preprocess_pipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      ],
    });

    const settings_bind_group = device.createBindGroup({
      label: 'preprocess render settings',
      layout: preprocess_pipeline.getBindGroupLayout(3),
      entries: [{ binding: 0, resource: { buffer: render_settings_buffer } }],
    });

    preprocess_bind_groups = [camera_bind_group, gaussian_bind_group, sort_bind_group, settings_bind_group];
  };
  build_preprocess(C.histogram_wg_size);


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: { module: render_shader, entryPoint: 'vs_main' },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        // back-to-front "over" blending with premultiplied alpha:
        // dst = src + (1 - src.a) * dst
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });

  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      // after 4 radix passes the sorted result ends up back in ping_pong[0]
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });


  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    // timestamps 0/1 = preprocess begin/end
    const pass = encoder.beginComputePass({ label: 'preprocess', timestampWrites: timer.writes(0, 1) });
    pass.setPipeline(preprocess_pipeline);
    preprocess_bind_groups.forEach((g, i) => pass.setBindGroup(i, g));
    const [x, y] = dispatch_size(pc.num_points, workgroup_size);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    // timestamps 2/3 = render begin/end (sort time = t2 - t1)
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        clearValue: [0, 0, 0, 1],
        storeOp: 'store',
      }],
      timestampWrites: timer.writes(2, 3),
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };


  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // reset counters every frame: keys_size = 0, dispatch_x = 0
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

      preprocess(encoder);
      sorter.sort(encoder);
      // keys_size (offset 0 of sort_info) -> instanceCount (offset 4 of indirect args)
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);
      render(encoder, texture_view);
      timer.resolve(encoder, sorter.sort_info_buffer);   // count = visible splats
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