/** CPU-only checks of series publication: first file at once, then file order with a bounded look-ahead. */
import assert from "node:assert/strict";
import test from "node:test";
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import { RansResidentSeries } from "../../src/quantem/gpu/detector/webgpu/rans-series";
import type { QemByteFile } from "../../src/quantem/gpu/detector/webgpu/qem-source";

const settle = async () => { for (let turn = 0; turn < 10; turn++) await new Promise(resolve => setTimeout(resolve, 0)); };
const device = {} as GPUDevice;
const files = (count: number) => Array.from({ length: count }, (_, index) => ({ name: `t${index}.qem`, size: 0 }) as unknown as QemByteFile);

type FakeSet = { name: string; disposed: boolean };
/** Replace loadQemFile with loads that the test resolves or rejects by file name. */
function controlledLoads() {
  const loads = new Map<string, { resolve: (shape?: number[]) => void; reject: (error: Error) => void }>();
  const sets: FakeSet[] = [];
  (RansResidentSet as unknown as { loadQemFile: unknown }).loadQemFile = (_device: GPUDevice, file: QemByteFile) =>
    new Promise((resolve, reject) => loads.set(file.name, {
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
  await settle();
  control.loads.get("t0.qem")!.resolve();
  const series = await loading;
  return { ...control, series, published };
}

test("the first acquisition is published alone, the rest in file order with three loads ahead", async () => {
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
    assert.deepEqual(sets.filter(set => !set.disposed).map(set => set.name), stop === "dispose" ? [] : ["t0.qem"]);
    series.dispose();
  }
});
