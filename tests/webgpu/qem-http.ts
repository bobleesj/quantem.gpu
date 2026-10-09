/** CPU-only checks of .qem admission from files served by a Range-capable HTTP server. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { qemFileSource, qemHttpFiles } from "../../src/quantem/gpu/detector/webgpu/qem-source";

const fixture = new Uint8Array(readFileSync("tests/data/qem-v2/u16-multiple-chunks.qem"));
Object.assign(globalThis, { location: { href: "http://localhost/viewer/index.html" } });

type Request = { url: string; method: string; cache?: string; range?: string };

/** Answer fetch like a static server: HEAD gives the size, GET honors one byte range. */
function serve(bytes: Uint8Array, options: { status?: number; ignoreRange?: boolean } = {}): Request[] {
  const requests: Request[] = [];
  globalThis.fetch = (async (url: URL, init: RequestInit = {}) => {
    const range = (init.headers as Record<string, string> | undefined)?.Range;
    requests.push({ url: String(url), method: init.method ?? "GET", cache: init.cache, range });
    if (options.status) return new Response(null, { status: options.status });
    if (init.method === "HEAD") return new Response(null, { headers: { "Content-Length": String(bytes.length) } });
    if (!range || options.ignoreRange) return new Response(bytes.slice(), { status: 200 });
    const [first, last] = /bytes=(\d+)-(\d+)/.exec(range)!.slice(1).map(Number);
    return new Response(bytes.slice(first, last + 1), { status: 206 });
  }) as typeof fetch;
  return requests;
}

test("served files are read through uncached ranges and admit the same streams as a local File", async () => {
  const requests = serve(fixture);
  const [file] = await qemHttpFiles("data/", ["u16-multiple-chunks.qem"]);
  assert.equal(file.size, fixture.length);
  const served = await qemFileSource(file);
  const local = await qemFileSource(new File([fixture], "u16-multiple-chunks.qem"));
  const manifest = JSON.parse(new TextDecoder().decode(await local.read("manifest.json")));
  const blocks = manifest.tilts[0].blocks_meta as { index: number; byte_start: number; byte_end: number }[];
  assert.ok(blocks.length > 1);
  for (const name of ["manifest.json", "entries", ...blocks.flatMap(block => [`columns-${block.index}`, `t0-offsets-${block.index}.u32`])]) {
    assert.deepEqual(new Uint8Array(await served.read(name)), new Uint8Array(await local.read(name)), name);
  }
  for (const block of blocks) {
    assert.deepEqual(new Uint8Array(await served.read("payload", block.byte_start, block.byte_end)), fixture.subarray(block.byte_start, block.byte_end));
  }
  assert.equal(requests[0].method, "HEAD");
  assert.ok(requests.every(request => request.url === "http://localhost/viewer/data/u16-multiple-chunks.qem"));
  assert.ok(requests.every(request => request.cache === "no-store"), "every request bypasses the HTTP cache");
  assert.ok(requests.slice(1).every(request => request.method === "GET" && request.range));
});

test("a server that ignores byte ranges is rejected", async () => {
  serve(fixture, { ignoreRange: true });
  const [file] = await qemHttpFiles("data/", ["u16-multiple-chunks.qem"]);
  await assert.rejects(qemFileSource(file), /ignored a byte range/);
});

test("missing files and names outside the served folder are rejected", async () => {
  serve(fixture, { status: 404 });
  await assert.rejects(qemHttpFiles("data/", ["missing.qem"]), /cannot open missing\.qem/);
  serve(fixture);
  for (const name of ["", ".", "..", "../other.qem", "sub/other.qem", "sub\\other.qem"]) {
    await assert.rejects(qemHttpFiles("data/", [name]), /invalid QEM file name/);
  }
});
