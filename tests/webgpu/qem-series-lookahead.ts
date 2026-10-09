/** CPU-only checks of series publication: first file at once, then file order with a bounded look-ahead. */
import assert from "node:assert/strict";
import test from "node:test";
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import { RansResidentSeries } from "../../src/quantem/gpu/detector/webgpu/rans-series";
import type { QemByteFile } from "../../src/quantem/gpu/detector/webgpu/qem-source";
import { syntheticQem } from "./qem-synthetic";

const settle = async () => { for (let turn = 0; turn < 10; turn++) await new Promise(resolve => setTimeout(resolve, 0)); };
const device = {} as GPUDevice;
// Real headers: the series reads every one before it starts loading.
const acquisition = syntheticQem([[64]]);
const files = (count: number) => Array.from({ length: count }, (_, index) => new File([acquisition], `t${index}.qem`) as QemByteFile);
const started = async (loads: Map<string, unknown>, name: string) => { while (!loads.has(name)) await new Promise(resolve => setTimeout(resolve, 0)); };

type FakeSet = { name: string; disposed: boolean };
/** Replace loadQemFile with loads that the test resolves or rejects by file name. */
function controlledLoads() {
  const loads = new Map<string, { resolve: (shape?: number[]) => void; reject: (error: Error) => void; signal?: AbortSignal }>();
  const sets: FakeSet[] = [];
  (RansResidentSet as unknown as { loadQemFile: unknown }).loadQemFile = (_device: GPUDevice, file: QemByteFile, _status: unknown, _badPixels: unknown, signal?: AbortSignal) =>
    new Promise((resolve, reject) => loads.set(file.name, { signal,
      resolve: (shape = [1, 2, 3, 4]) => {
        const set = { name: file.name, disposed: false, shape, nativeDtype: "uint16", badPx: new Uint32Array(0),
          computes: [{ name: file.name }], dispose() { this.disposed = true; } };
        sets.push(set);
        resolve(set);
      },
      reject,
    }));
  return { loads, sets, started: () => [...loads.keys()] };
}

async function loadSeries(count: number, controller = new AbortController()) {
  const control = controlledLoads();
  const published: number[] = [];
  const loading = RansResidentSeries.load(device, files(count), () => {}, series => published.push(series.loadedAcquisitions), controller.signal);
  await started(control.loads, "t0.qem");
  control.loads.get("t0.qem")!.resolve();
  const series = await loading;
  return { ...control, series, published };
}

test("the first acquisition is published alone, the rest in file order with three loads in flight", async () => {
  const { loads, series, published, started } = await loadSeries(6);
  assert.deepEqual(published, [1]);
  await settle();
  assert.deepEqual(started(), ["t0.qem", "t1.qem", "t2.qem", "t3.qem"]);
  loads.get("t3.qem")!.resolve();
  loads.get("t2.qem")!.resolve();
  await settle();
  assert.deepEqual(published, [1], "later files wait for the next one in order");
  loads.get("t1.qem")!.resolve();
  await settle();
  assert.deepEqual(published, [1, 2, 3, 4]);
  assert.deepEqual(started(), ["t0.qem", "t1.qem", "t2.qem", "t3.qem", "t4.qem", "t5.qem"]);
  loads.get("t5.qem")!.resolve();
  loads.get("t4.qem")!.resolve();
  await series.completion;
  assert.deepEqual(published, [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(series.computes.map(compute => (compute as unknown as { name: string }).name), files(6).map(file => file.name));
});

test("a failed or mismatched acquisition stops the series and releases every unpublished set", async () => {
  for (const failure of ["error", "mismatch"]) {
    const { loads, sets, series, published } = await loadSeries(5);
    await settle();
    loads.get("t2.qem")!.resolve();
    loads.get("t3.qem")!.resolve();
    if (failure === "error") loads.get("t1.qem")!.reject(new Error("payload checksum mismatch"));
    else loads.get("t1.qem")!.resolve([1, 2, 3, 5]);
    await assert.rejects(series.completion, failure === "error" ? /checksum mismatch/ : /differs from the first file/);
    assert.deepEqual(published, [1]);
    assert.deepEqual(sets.filter(set => set.disposed).map(set => set.name).sort(), failure === "error" ? ["t2.qem", "t3.qem"] : ["t1.qem", "t2.qem", "t3.qem"]);
    series.dispose();
    assert.ok(sets.every(set => set.disposed));
  }
});

test("disposal ends quietly and cancellation rejects, releasing loads that finish afterwards", async () => {
  for (const stop of ["dispose", "abort"]) {
    const controller = new AbortController();
    const { loads, sets, series } = await loadSeries(4, controller);
    await settle();
    if (stop === "dispose") series.dispose(); else controller.abort();
    loads.get("t1.qem")!.reject(new Error("late failure"));
    for (const name of ["t2.qem", "t3.qem"]) loads.get(name)!.resolve();
    if (stop === "dispose") await series.completion;
    else await assert.rejects(series.completion, { name: "AbortError" });
    await settle();
    assert.deepEqual(sets.filter(set => !set.disposed).map(set => set.name), stop === "dispose" ? [] : ["t0.qem"]);
    series.dispose();
  }
});

test("disposal and cancellation stop loads in flight instead of waiting for them", async () => {
  for (const stop of ["dispose", "abort"]) {
    const controller = new AbortController();
    const { loads, sets, series } = await loadSeries(4, controller);
    await settle();
    const inFlight = ["t1.qem", "t2.qem", "t3.qem"].map(name => loads.get(name)!);
    assert.ok(inFlight.every(load => load.signal && !load.signal.aborted));
    if (stop === "dispose") series.dispose(); else controller.abort();
    // The loads never finish on their own: completion must not wait for them.
    const settled = await Promise.race([series.completion.then(() => "resolved", error => error.name), new Promise(resolve => setTimeout(() => resolve("waiting"), 200))]);
    assert.equal(settled, stop === "dispose" ? "resolved" : "AbortError");
    assert.ok(inFlight.every(load => load.signal!.aborted), "every load in flight is told to stop");
    inFlight.forEach(load => load.resolve());
    await settle();
    assert.deepEqual(sets.filter(set => !set.disposed).map(set => set.name), stop === "dispose" ? [] : ["t0.qem"]);
    series.dispose();
  }
});

test("a progress callback that throws releases the first acquisition", async () => {
  const { loads, sets } = controlledLoads();
  const loading = RansResidentSeries.load(device, files(3), () => {}, () => { throw new Error("panel failed"); }, new AbortController().signal);
  await started(loads, "t0.qem");
  loads.get("t0.qem")!.resolve();
  await assert.rejects(loading, /panel failed/);
  assert.deepEqual(sets.map(set => [set.name, set.disposed]), [["t0.qem", true]]);
  await settle();
  assert.deepEqual([...loads.keys()], ["t0.qem"], "no other acquisition starts loading");
});

test("a series whose headers disagree fails before any acquisition loads", async () => {
  const { loads } = controlledLoads();
  const mismatched = [...files(2), new File([syntheticQem([[64], [64]])], "t2.qem") as QemByteFile];
  const outcome = await Promise.race([
    RansResidentSeries.load(device, mismatched, () => {}, () => {}, new AbortController().signal).then(() => "loaded", error => String(error)),
    new Promise(resolve => setTimeout(() => resolve("waiting on the first load"), 500)),
  ]);
  assert.match(String(outcome), /t2\.qem has shape 1x1024x1x1/);
  assert.equal(loads.size, 0, "no payload of any file was read");
});
