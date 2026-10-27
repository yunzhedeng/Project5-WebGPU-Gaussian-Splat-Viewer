import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { default as get_renderer_gaussian_tile } from './gaussian-tile-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';
import { GpuTimer } from '../utils/gpu-timer';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
  timer?: GpuTimer,   // GPU timestamps + visible splat count (performance analysis)
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer, gaussian_tile?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let gaussian_tile_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: Renderer | undefined; 
  let renderer: Renderer | undefined; 
  let cameras;
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);
  
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });
  

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
    // performance analysis
    gpu_preprocess_ms: 0.0,
    gpu_sort_ms: 0.0,
    gpu_render_ms: 0.0,
    gpu_total_ms: 0.0,
    visible_splats: 0,
    frustum_culling: true,
    workgroup_size: 256,
  };
  const gaussian_renderers = () =>
    [gaussian_renderer, gaussian_tile_renderer].filter(r => r !== undefined) as GaussianRenderer[];

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
    });
  }
  {
    // GPU timing needs the 'timestamp-query' feature (see main.ts)
    const perf = pane.addFolder({ title: device.features.has('timestamp-query')
      ? 'GPU time (ms)' : 'GPU time (timestamp-query unavailable)' });
    perf.addMonitor(params, 'gpu_preprocess_ms', { label: 'preprocess', format: (v: number) => v.toFixed(3) });
    perf.addMonitor(params, 'gpu_sort_ms', { label: 'sort', format: (v: number) => v.toFixed(3) });
    perf.addMonitor(params, 'gpu_render_ms', { label: 'render', format: (v: number) => v.toFixed(3) });
    perf.addMonitor(params, 'gpu_total_ms', { label: 'total', format: (v: number) => v.toFixed(3) });
    perf.addMonitor(params, 'visible_splats', { label: 'visible', format: (v: number) => v.toFixed(0) });
    perf.addInput(params, 'frustum_culling').on('change', (e) => {
      gaussian_renderers().forEach(r => r.set_frustum_culling(e.value));
    });
    perf.addInput(params, 'workgroup_size', {
      options: { '16': 16, '32': 32, '64': 64, '128': 128, '256': 256 },
    }).on('change', (e) => {
      gaussian_renderers().forEach(r => r.set_workgroup_size(e.value));
    });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
        gaussian_tile: 'gaussian_tile',
      }
    }).on('change', (e) => {
      renderer = renderers[e.value];
    });
  }
  {
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        const pc = await load(uploadedFile, device);
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer.set_gaussian_multiplier(params.gaussian_multiplier);
        gaussian_tile_renderer = get_renderer_gaussian_tile(pc, device, presentation_format, camera.uniform_buffer, canvas);
        gaussian_tile_renderer.set_gaussian_multiplier(params.gaussian_multiplier);
        gaussian_renderers().forEach(r => {
          r.set_frustum_culling(params.frustum_culling);
          r.set_workgroup_size(params.workgroup_size);
        });
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
          gaussian_tile: gaussian_tile_renderer,
        };
        renderer = renderers[params.renderer];
        ply_file_loaded = true;
      }else{
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        cameras=await load_camera_presets(file.value);
        camera.set_preset(cameras[0]);
        cam_file_loaded = true;
      }else{
        cam_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(
      params,
      'gaussian_multiplier',
      {min: 0, max: 1.5}
    ).on('change', (e) => {
      //TODO: Bind constants to the gaussian renderer.
      gaussian_renderer?.set_gaussian_multiplier(e.value);
      gaussian_tile_renderer?.set_gaussian_multiplier(e.value);
    });
  }

  document.addEventListener('keydown', (event) => {
    switch(event.key) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        const i = parseInt(event.key);
        if (!cameras || i >= cameras.length) break;
        console.log(`set to camera preset ${i}`);
        camera.set_preset(cameras[i]);
        break;
    }
  });

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);
      if (renderer.timer) {
        renderer.timer.after_submit();
        const s = renderer.timer.stats;
        params.gpu_preprocess_ms = s.preprocess_ms;
        params.gpu_sort_ms = s.sort_ms;
        params.gpu_render_ms = s.render_ms;
        params.gpu_total_ms = s.total_ms;
        params.visible_splats = s.count;
      }
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}