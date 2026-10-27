/// <reference types="@webgpu/types" />
export interface GpuStats {
  preprocess_ms: number,
  sort_ms: number,
  render_ms: number,
  total_ms: number,
  count: number,
}

interface Readback { buffer: GPUBuffer, state: 'free' | 'copied' | 'mapping', has_count: boolean }

const READBACK_SIZE = 4 * 8 + 4;   // 4 x u64 timestamps + 1 x u32 counter

export class GpuTimer {
  readonly enabled: boolean;
  readonly stats: GpuStats = { preprocess_ms: 0, sort_ms: 0, render_ms: 0, total_ms: 0, count: 0 };

  private query_set?: GPUQuerySet;
  private resolve_buffer?: GPUBuffer;
  private readbacks: Readback[] = [];
  private pending?: Readback;

  // render_only: a single pass (point cloud) writes query 0 at begin and 3 at end
  constructor(private device: GPUDevice, private render_only = false) {
    this.enabled = device.features.has('timestamp-query');
    if (!this.enabled) return;
    this.query_set = device.createQuerySet({ type: 'timestamp', count: 4 });
    this.resolve_buffer = device.createBuffer({
      label: 'timestamp resolve', size: 4 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    for (let i = 0; i < 3; i++) {
      this.readbacks.push({
        buffer: device.createBuffer({
          label: 'timestamp readback', size: READBACK_SIZE,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        state: 'free',
        has_count: false,
      });
    }
  }

  // timestampWrites for a pass: begin/end are query indices 0..3 (or undefined)
  writes(begin?: number, end?: number): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled) return undefined;
    const w: GPUComputePassTimestampWrites = { querySet: this.query_set! };
    if (begin !== undefined) w.beginningOfPassWriteIndex = begin;
    if (end !== undefined) w.endOfPassWriteIndex = end;
    return w;
  }

  // call at the end of frame encoding
  resolve(encoder: GPUCommandEncoder, count_buffer?: GPUBuffer) {
    if (!this.enabled) return;
    const rb = this.readbacks.find(r => r.state === 'free');
    if (!rb) return;   // all readbacks busy, skip this frame
    encoder.resolveQuerySet(this.query_set!, 0, 4, this.resolve_buffer!, 0);
    encoder.copyBufferToBuffer(this.resolve_buffer!, 0, rb.buffer, 0, 4 * 8);
    if (count_buffer) encoder.copyBufferToBuffer(count_buffer, 0, rb.buffer, 4 * 8, 4);
    rb.state = 'copied';
    rb.has_count = !!count_buffer;
    this.pending = rb;
  }

  // call right after queue.submit
  after_submit() {
    const rb = this.pending;
    if (!rb || rb.state !== 'copied') return;
    this.pending = undefined;
    rb.state = 'mapping';
    rb.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = rb.buffer.getMappedRange();
      const t = new BigInt64Array(data, 0, 4);
      const count = new Uint32Array(data, 4 * 8, 1)[0];
      const count_buffer_used = rb.has_count;
      const ms = (a: bigint, b: bigint) => Math.max(0, Number(b - a)) / 1e6;
      // exponential moving average to make the numbers readable
      const k = 0.2, s = this.stats;
      if (this.render_only) {
        s.render_ms += k * (ms(t[0], t[3]) - s.render_ms);
      } else {
        s.preprocess_ms += k * (ms(t[0], t[1]) - s.preprocess_ms);
        s.sort_ms += k * (ms(t[1], t[2]) - s.sort_ms);
        s.render_ms += k * (ms(t[2], t[3]) - s.render_ms);
      }
      s.total_ms += k * (ms(t[0], t[3]) - s.total_ms);
      if (count_buffer_used) s.count = count;
      rb.buffer.unmap();
      rb.state = 'free';
    }).catch(() => { rb.state = 'free'; });
  }
}