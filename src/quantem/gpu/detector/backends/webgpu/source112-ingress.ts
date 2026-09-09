/** Ordered payload ingress after record authentication. Owns a bounded four-buffer staging ring. */
export interface Source112IngressProfile {
  payloadHostCopyMs: number;
  payloadMapPendingMs: number;
  payloadMapBlockedMs: number;
  payloadCopySubmitMs: number;
  payloadFenceWaitMs: number;
  payloadChunks: number;
}
type Outcome = { error?: unknown };
type Stage = { buffer: GPUBuffer; ready?: Promise<Outcome> };
const STAGE_BYTES = 32 * 1024 * 1024;
const STAGES = 4;

export class Source112Ingress {
  private stages: Stage[] = [];
  private pending = new Set<Promise<Outcome>>();
  private disposed = false;
  private busy = false;
  private cursor = 0;
  private lost: GPUDeviceLostInfo | null = null;
  private cleanup?: Promise<void>;

  constructor(private device: GPUDevice, private profile: Source112IngressProfile,
    private ownedBytes: (bytes: number) => void, private signal?: AbortSignal) {
    signal?.throwIfAborted();
    if (device.limits.maxBufferSize < STAGE_BYTES) throw Error('Payload ingress requires a 32 MiB staging buffer.');
    void device.lost.then(info => { this.lost = info; });
    try {
      for (let i = 0; i < STAGES; i++) {
        this.stages.push({ buffer: device.createBuffer({ size: STAGE_BYTES,
          usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true }) });
        ownedBytes(this.stages.length * STAGE_BYTES);
      }
    } catch (error) {
      for (const stage of this.stages) stage.buffer.destroy();
      ownedBytes(0); throw error;
    }
  }

  private alive() {
    this.signal?.throwIfAborted();
    if (this.disposed) throw Error('Payload ingress is closed.');
    if (this.lost) throw Error(`Payload device lost: ${this.lost.message}`);
  }

  private async ready(stage: Stage) {
    if (stage.ready) {
      const began = performance.now(), result = await stage.ready;
      this.profile.payloadMapBlockedMs += performance.now() - began;
      stage.ready = undefined;
      if ('error' in result) throw result.error;
    }
    this.alive();
  }

  /** Enqueue one authenticated record; staging reuse waits only for its own copy.
   * The input may be released on return. Call finish before exposing the source. */
  async write(destination: GPUBuffer, offset: number, raw: ArrayBuffer) {
    this.alive();
    if (this.busy) throw Error('Await the current record before uploading another.');
    if (!Number.isSafeInteger(offset) || offset < 0 || offset % 4 || raw.byteLength % 4
      || offset + raw.byteLength > destination.size) throw Error('Payload copy requires an aligned in-bounds complete record.');
    this.busy = true;
    try {
      const input = new Uint8Array(raw);
      for (let begin = 0; begin < raw.byteLength;) {
        const stage = this.stages[this.cursor++ % STAGES]; await this.ready(stage);
        const bytes = Math.min(STAGE_BYTES, raw.byteLength - begin);
        let began = performance.now();
        new Uint8Array(stage.buffer.getMappedRange()).set(input.subarray(begin, begin + bytes));
        this.profile.payloadHostCopyMs += performance.now() - began;
        stage.buffer.unmap();
        began = performance.now();
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(stage.buffer, 0, destination, offset + begin, bytes);
        this.device.queue.submit([encoder.finish()]);
        this.profile.payloadCopySubmitMs += performance.now() - began;
        this.profile.payloadChunks++;
        const requested = performance.now();
        // Handle rejection immediately while the other ring slot may still be in use.
        const mapping = stage.buffer.mapAsync(GPUMapMode.WRITE);
        const outcome: Promise<Outcome> = mapping.then(() => ({}), error => ({ error })).then(result => {
          this.profile.payloadMapPendingMs += performance.now() - requested;
          this.pending.delete(outcome); return result;
        });
        this.pending.add(outcome); stage.ready = outcome;
        begin += bytes;
      }
    } finally { this.busy = false; }
  }

  /** Complete all authenticated copies before their owner is made available. */
  async finish() {
    this.alive();
    if (this.busy) throw Error('Await the current record before finishing ingress.');
    const began = performance.now();
    try { await this.device.queue.onSubmittedWorkDone(); }
    finally { this.profile.payloadFenceWaitMs += performance.now() - began; }
    for (const stage of this.stages) await this.ready(stage);
    this.alive();
  }

  /** Cancel outstanding maps, drain handled promises, then release only owned staging. */
  dispose(): Promise<void> {
    if (this.cleanup) return this.cleanup;
    this.disposed = true;
    // Destroying a buffer cancels an outstanding map; do this before awaiting it.
    for (const stage of this.stages) stage.buffer.destroy();
    this.ownedBytes(0);
    this.cleanup = (async () => {
      await Promise.allSettled([...this.pending]);
      await this.device.queue.onSubmittedWorkDone();
      this.stages = [];
    })();
    return this.cleanup;
  }
}
