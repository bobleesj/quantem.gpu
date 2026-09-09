/** Bounded file reads and authentication off the browser's presentation thread. */
export interface AuthenticatedRead {
  raw: ArrayBuffer;
  digest: string;
  readMs: number;
  hashMs: number;
}
const WORKER_SOURCE = `self.onmessage = async event => {
  try {
    const began = performance.now();
    const raw = await event.data.arrayBuffer();
    const readMs = performance.now() - began;
    const hashBegan = performance.now();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', raw))]
      .map(value => value.toString(16).padStart(2, '0')).join('');
    self.postMessage({raw, digest, readMs, hashMs: performance.now() - hashBegan}, [raw]);
  } catch (error) { self.postMessage({error: String(error)}); }
};`;
interface Lane {
  worker: Worker;
  job?: {resolve: (result: AuthenticatedRead) => void; reject: (error: unknown) => void};
}

/** The ordered prefetcher owns the 2 GiB reservation and limits concurrency. */
export class Source112WorkerReader {
  private readonly lanes: Lane[] = [];
  private readonly url = URL.createObjectURL(new Blob([WORKER_SOURCE], {type: 'text/javascript'}));
  private closed = false;
  private readonly abort = () => this.dispose(this.signal?.reason);
  constructor(private readonly signal?: AbortSignal) {
    signal?.addEventListener('abort', this.abort, {once: true});
    if (signal?.aborted) this.abort();
  }
  read(blob: Blob): Promise<AuthenticatedRead> {
    this.signal?.throwIfAborted();
    if (this.closed) throw Error('Source reader is closed; select the folder again.');
    let lane = this.lanes.find(lane => !lane.job);
    if (!lane) {
      if (this.lanes.length === 8) throw Error('Source reader exceeds its eight-record prefetch limit.');
      const worker = new Worker(this.url);
      lane = {worker}; this.lanes.push(lane);
      const selected = lane;
      worker.onmessage = event => {
        const job = selected.job;
        selected.job = undefined;
        if (event.data.error) job?.reject(Error(event.data.error));
        else job?.resolve(event.data as AuthenticatedRead);
      };
      worker.onerror = event => this.dispose(Error(event.message));
      worker.onmessageerror = () => this.dispose(Error('Could not receive encoded file bytes from the reader.'));
    }
    const selected = lane;
    return new Promise((resolve, reject) => {
      selected.job = {resolve, reject};
      try { selected.worker.postMessage(blob); }
      catch (error) { selected.job = undefined; reject(error); }
    });
  }
  dispose(reason: unknown = new Error('Encoded file reading stopped.')) {
    if (this.closed) return;
    this.closed = true;
    this.signal?.removeEventListener('abort', this.abort);
    for (const lane of this.lanes) {
      lane.worker.terminate();
      lane.job?.reject(reason);
      lane.job = undefined;
    }
    this.lanes.length = 0;
    URL.revokeObjectURL(this.url);
  }
}
