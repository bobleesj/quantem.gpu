/** Ordered .qem acquisitions, each authenticated and decoded into its own resident set.
 *
 * The first acquisition is published as soon as it is ready. The others load
 * behind it with a bounded look-ahead and are published strictly in file order,
 * so a viewer shows and compares panels while the rest of the series loads.
 */
import { RansResidentSet, type RansDetectorCompute, type RansLoadProfile } from "./rans";
import type { QemByteFile } from "./qem-source";

// Acquisitions loading at once, the next one to publish included: file
// authentication overlaps the decoder build of earlier files while host
// memory stays bounded.
const LOOKAHEAD = 3;

export class RansResidentSeries {
  /** Filled in file order; entries from loadedAcquisitions on are not loaded yet. */
  readonly computes: RansDetectorCompute[];
  readonly acquisitionCount: number;
  readonly acquisitionMode = "local-folder" as const;
  readonly shape: RansResidentSet["shape"];
  readonly nativeDtype: RansResidentSet["nativeDtype"];
  readonly badPx: Uint32Array;
  /** Resolves once every acquisition is published or the series is disposed;
   * rejects with the first failure, or with an AbortError when cancelled. */
  readonly completion: Promise<void>;
  loadedAcquisitions = 1;
  private readonly sets: RansResidentSet[];
  private disposed = false;
  private finishedAt: number;

  private constructor(first: RansResidentSet, files: QemByteFile[], readonly device: GPUDevice,
    private readonly started: number, status: (text: string) => void,
    progress: (series: RansResidentSeries) => void, signal: AbortSignal, badPixels: number[]) {
    this.acquisitionCount = files.length;
    this.computes = new Array(files.length);
    this.computes[0] = first.computes[0];
    this.sets = [first];
    this.shape = first.shape; this.nativeDtype = first.nativeDtype; this.badPx = first.badPx;
    this.finishedAt = performance.now();
    progress(this);
    this.completion = this.publishRemaining(first, files, status, progress, signal, badPixels);
  }

  /** Load the remaining acquisitions with bounded look-ahead and publish them in file order. */
  private async publishRemaining(first: RansResidentSet, files: QemByteFile[], status: (text: string) => void,
    progress: (series: RansResidentSeries) => void, signal: AbortSignal, badPixels: number[]): Promise<void> {
    type Loaded = { set: RansResidentSet } | { error: unknown };
    const pending = new Map<number, Promise<Loaded>>();
    let frontier = 1;
    const begin = (index: number) => {
      if (index >= files.length || pending.has(index) || this.disposed || signal.aborted) return;
      pending.set(index, RansResidentSet.loadQemFile(this.device, files[index], text => {
        // Only the next acquisition to publish reports its progress.
        if (index === frontier && !this.disposed && !signal.aborted) status(`${index + 1}/${files.length} ${files[index].name}: ${text}`);
      }, badPixels).then(set => ({ set }), error => ({ error })));
    };
    try {
      // Let the first panel paint before more loads compete for the device.
      await new Promise(resolve => setTimeout(resolve, 0));
      for (let index = 1; index <= LOOKAHEAD; index++) begin(index);
      for (let index = 1; index < files.length; index++) {
        frontier = index;
        if (this.disposed) return;
        signal.throwIfAborted();
        const loaded = await pending.get(index)!;
        pending.delete(index);
        // Teardown ends quietly and cancellation with an AbortError, whichever
        // step they interrupt; a load that finishes afterwards is only released.
        if (this.disposed || signal.aborted) {
          if ("set" in loaded) loaded.set.dispose();
          if (this.disposed) return;
          signal.throwIfAborted();
        }
        if ("error" in loaded) throw loaded.error;
        const next = loaded.set;
        if (JSON.stringify(next.shape) !== JSON.stringify(first.shape) || next.nativeDtype !== first.nativeDtype || JSON.stringify([...next.badPx]) !== JSON.stringify([...first.badPx])) {
          next.dispose();
          throw new Error(`Acquisition ${files[index].name} differs from the first file's geometry, dtype or detector validity mask; open it separately.`);
        }
        this.sets.push(next); this.computes[index] = next.computes[0];
        this.loadedAcquisitions = index + 1; this.finishedAt = performance.now();
        progress(this);
        begin(index + LOOKAHEAD);
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      status("");
    } finally {
      // A load cannot be interrupted mid-authentication: wait for each one and
      // release every unpublished set on cancellation, disposal or failure.
      for (const loaded of await Promise.all(pending.values())) if ("set" in loaded) loaded.set.dispose();
    }
  }

  /** Publish the first acquisition, then keep loading the others in order behind it. */
  static async load(device: GPUDevice, files: QemByteFile[], status: (text: string) => void,
    progress: (series: RansResidentSeries) => void, signal: AbortSignal, badPixels: number[] = []): Promise<RansResidentSeries> {
    if (!files.length) throw new Error("Select at least one .qem acquisition.");
    const started = performance.now();
    const first = await RansResidentSet.loadQemFile(device, files[0], text => status(`1/${files.length} ${files[0].name}: ${text}`), badPixels);
    if (signal.aborted) { first.dispose(); signal.throwIfAborted(); }
    return new RansResidentSeries(first, files, device, started, status, progress, signal, badPixels);
  }

  get payloadBytes() { return this.sets.reduce((sum, set) => sum + set.payloadBytes, 0); }
  get loadMs() { return this.sets.reduce((sum, set) => sum + set.loadMs, 0); }
  get checkpointMs() { return this.sets.reduce((sum, set) => sum + set.checkpointMs, 0); }
  /** Time from the start of loading to the latest published acquisition. */
  get readyMs() { return this.finishedAt - this.started; }
  get loadProfile(): RansLoadProfile {
    const total: RansLoadProfile = { metadataReadMs: 0, payloadReadMs: 0, payloadReadWaitMs: 0, payloadStageMs: 0, payloadChunks: 0, payloadBuffers: 0 };
    for (const set of this.sets) for (const key of Object.keys(total) as (keyof RansLoadProfile)[]) total[key] += set.loadProfile[key];
    return total;
  }
  readImage(tilt: number) { return this.sets[tilt].readImage(0); }
  readImageU32(tilt: number) { return this.sets[tilt].readImageU32(0); }

  /** Normalize display copies of several acquisitions in one submission; `tilts[i]` owns `buffers[i]`. */
  normalizeDisplayBuffers(buffers: GPUBuffer[], maskArea: number, tilts: number[]): void {
    const encoder = this.device.createCommandEncoder();
    buffers.forEach((buffer, index) => this.sets[tilts[index]].normalizeDisplayBuffers([buffer], maskArea, encoder));
    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void { this.disposed = true; for (const set of this.sets) set.dispose(); }
}
