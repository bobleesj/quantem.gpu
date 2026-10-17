/** Bounded, transferable SHA-256 jobs for QEM admission; no unverified bytes are staged. */
type Digest = { bytes: ArrayBuffer; hash: ArrayBuffer };
type Job = { resolve: (value: Digest) => void; reject: (error: unknown) => void };

const workerScript = `
onmessage = async ({data}) => {
  try {
    const hash = await crypto.subtle.digest("SHA-256", data.bytes);
    postMessage({id: data.id, bytes: data.bytes, hash}, [data.bytes, hash]);
  } catch (error) {
    postMessage({id: data.id, error: String(error)});
  }
};
postMessage({ready: true});
`;

/** Hash off the UI thread when workers are allowed, otherwise retain WebCrypto.
 *
 * Admission bounds calls to four chunks. Transfer ownership in both directions
 * instead of cloning 64 MiB chunks; callers must use the returned buffer. Workers
 * belong to this admission only and are terminated on completion or cancellation.
 */
export function createQemHasher(count: number, signal: AbortSignal): {
  digest(bytes: ArrayBuffer): Promise<Digest>;
  dispose(): void;
} {
  const workers: Worker[] = [];
  const jobs = new Map<number, Job>();
  const starting = new Set<(error: unknown) => void>();
  let next = 0, closed = false;
  let failure: unknown;
  const stop = (error: unknown) => {
    closed = true;
    failure = error;
    for (const reject of starting) reject(error);
    starting.clear();
    for (const job of jobs.values()) job.reject(error);
    jobs.clear();
    for (const worker of workers) worker.terminate();
    workers.length = 0;
  };
  const abort = () => stop(signal.reason);
  signal.addEventListener("abort", abort, { once: true });
  const ready = (async () => {
    if (count < 2 || typeof Worker === "undefined") return false;
    let url: string | undefined;
    try {
      url = URL.createObjectURL(new Blob([workerScript], { type: "text/javascript" }));
      const starts: Promise<void>[] = [];
      for (let index = 0; index < Math.min(count, 4); index++) {
        const worker = new Worker(url, { name: "qem-sha256" });
        workers.push(worker);
        const start = new Promise<void>((resolve, reject) => {
          let started = false;
          const timer = setTimeout(() => failed(new Error("QEM checksum worker did not start")), 5000);
          const failed = (error: unknown) => {
            clearTimeout(timer);
            starting.delete(failed);
            if (started) stop(error); else reject(error);
          };
          starting.add(failed);
          worker.onerror = event => {
            event.preventDefault();
            failed(new Error("QEM checksum worker failed; reload the intact acquisition"));
          };
          worker.onmessageerror = () => failed(new Error("QEM checksum worker message failed"));
          worker.onmessage = ({ data }) => {
            if (data.ready) { started = true; clearTimeout(timer); starting.delete(failed); resolve(); return; }
            const job = jobs.get(data.id);
            if (!job) return;
            jobs.delete(data.id);
            if (data.error) job.reject(new Error(data.error));
            else job.resolve({ bytes: data.bytes, hash: data.hash });
          };
        });
        // A later constructor may throw before Promise.all attaches handlers.
        void start.catch(() => {});
        starts.push(start);
      }
      // Do not transfer any data until startup succeeds. A restrictive CSP can
      // then fall back without losing the input buffer or skipping its digest.
      await Promise.all(starts);
      return true;
    } catch (error) {
      for (const reject of starting) reject(error);
      starting.clear();
      for (const worker of workers) worker.terminate();
      workers.length = 0;
      return false;
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  })();
  return {
    async digest(bytes) {
      signal.throwIfAborted();
      const parallel = await ready;
      signal.throwIfAborted();
      if (closed) throw failure;
      if (!parallel) return { bytes, hash: await crypto.subtle.digest("SHA-256", bytes) };
      return new Promise<Digest>((resolve, reject) => {
        const id = next++;
        jobs.set(id, { resolve, reject });
        try { workers[id % workers.length].postMessage({ id, bytes }, [bytes]); }
        catch (error) { jobs.delete(id); reject(error); }
      });
    },
    dispose() {
      signal.removeEventListener("abort", abort);
      stop(new DOMException("QEM verification ended", "AbortError"));
    },
  };
}
