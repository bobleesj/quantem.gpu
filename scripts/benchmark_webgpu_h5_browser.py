#!/usr/bin/env python
"""Benchmark a browser-local Show4DSTEM HDF5 load through Chrome CDP.

This is a maintainer harness for exported QuantEM widget HTML. It drives the
real browser File API, waits for the WebGPU local-HDF5 load profile, optionally
checks corrected-frame checksums, and writes a JSON artifact without recording
local raw-data paths.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import websocket


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdp", default="http://127.0.0.1:9239")
    parser.add_argument("--html-url", required=True)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--master-name", default="master.h5")
    parser.add_argument("--data-template", default="data_{index:06d}.h5")
    parser.add_argument(
        "--block-index-template",
        help=(
            "Optional metadata-only QH5IDX sidecar template, for example "
            "'data_{index:06d}.qh5idx'. Files are added to the same browser "
            "file input as the HDF5 master/data files."
        ),
    )
    parser.add_argument("--data-files", type=int, default=27)
    parser.add_argument("--frames", type=int, default=262144)
    parser.add_argument(
        "--source-scan-shape",
        help=(
            "Optional source scan shape as ROWS,COLS for crop tests where the "
            "exported widget shape is the cropped output shape."
        ),
    )
    parser.add_argument(
        "--scan-region",
        help=(
            "Optional true source scan crop as r0,r1,c0,c1. The browser loader "
            "decodes only that frame window; this is not load-then-slice."
        ),
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--decode-batch", type=int)
    parser.add_argument("--det-bin", type=int, default=1)
    parser.add_argument(
        "--frame-low8",
        choices=("default", "true", "false"),
        default="default",
        help="Runtime override for __BSLZ4_FRAME_LOW8.",
    )
    parser.add_argument(
        "--u32-shared-low8",
        action="store_true",
        help="Use the experimental u32-per-byte workgroup-memory low8 decoder.",
    )
    parser.add_argument(
        "--single-parse-low8",
        action="store_true",
        help="Use the experimental single-lane-parser low8 decoder.",
    )
    parser.add_argument(
        "--frame-serial-low8",
        action="store_true",
        help="Use the experimental frame-level serial low8 decoder.",
    )
    parser.add_argument(
        "--frames-per-wg",
        type=int,
        help="Experimental number of diffraction patterns per frame-coop decoder workgroup.",
    )
    parser.add_argument(
        "--frame-wg",
        type=int,
        choices=(32, 64, 128),
        help="Experimental workgroup size for the frame-cooperative low8 decoder.",
    )
    parser.add_argument(
        "--no-pipeline-staging",
        action="store_true",
        help="Disable the default overlapped staging-upload/decode scheduler.",
    )
    parser.add_argument(
        "--upload",
        choices=("default", "staging", "write-buffer", "mapped", "combined"),
        default="default",
        help="Runtime override for the compressed-byte WebGPU upload route.",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--dpc-reps",
        type=int,
        default=0,
        help="Measure full no-bin DPC row/col browser paths after load.",
    )
    parser.add_argument(
        "--dpc-warmup",
        type=int,
        default=1,
        help="Warm DPC row/col GPU/readback caches before timing repeated phases.",
    )
    parser.add_argument(
        "--fps-ms",
        type=int,
        default=1000,
        help="requestAnimationFrame sampling window for DPC interaction checks.",
    )
    parser.add_argument("--checksum-json", type=Path)
    parser.add_argument(
        "--dpc-reference-json",
        type=Path,
        help="Optional local reference manifest with DPC/iDPC .f32 files for browser max/mean error checks.",
    )
    parser.add_argument(
        "--frame-index-json",
        type=Path,
        help="Optional metadata-only frame-offset manifest injected as __QT_H5_LOCAL_FRAME_INDEX.",
    )
    parser.add_argument(
        "--require-local-profile",
        action="store_true",
        help="Wait for a local-file load profile instead of accepting the URL/fetch fallback.",
    )
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _http_json(cdp: str, method: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{cdp}{path}", method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


class CdpTarget:
    def __init__(self, cdp: str, url: str):
        target = _http_json(cdp, "PUT", "/json/new?" + urllib.parse.quote(url, safe=""))
        self.target_id = str(target["id"])
        self._ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=20)
        self._next_id = 0

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
        self._next_id += 1
        msg_id = self._next_id
        self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self._ws.recv())
            if message.get("id") != msg_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            return message.get("result", {})
        raise TimeoutError(method)

    def eval(self, expression: str, *, timeout: float = 30, await_promise: bool = False) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
            timeout=timeout,
        )
        value = result.get("result", {})
        if value.get("subtype") == "error":
            raise RuntimeError(value)
        return value.get("value")

    def close(self, cdp: str) -> None:
        try:
            self.call("Target.closeTarget", {"targetId": self.target_id}, timeout=5)
        except Exception:
            pass
        self._ws.close()


def _runtime_prelude(args: argparse.Namespace) -> str:
    statements: list[str] = []
    if args.source_scan_shape:
        rows, cols = _parse_int_tuple(args.source_scan_shape, 2, "--source-scan-shape")
        statements.append(f"globalThis.__QT_H5_SOURCE_SCAN_ROWS={rows};")
        statements.append(f"globalThis.__QT_H5_SOURCE_SCAN_COLS={cols};")
    if args.scan_region:
        region = _parse_int_tuple(args.scan_region, 4, "--scan-region")
        statements.append(f"globalThis.__QT_H5_SCAN_REGION={json.dumps(region)};")
    if args.workers is not None:
        statements.append(f"globalThis.__QT_H5_LOCAL_WORKERS={int(args.workers)};")
    if args.group_size is not None:
        statements.append(f"globalThis.__QT_H5_LOCAL_GROUP={int(args.group_size)};")
    if args.decode_batch is not None:
        statements.append(f"globalThis.__QT_H5_DECODE_BATCH={int(args.decode_batch)};")
    if args.det_bin and int(args.det_bin) > 1:
        statements.append(f"globalThis.__QT_H5_DET_BIN={int(args.det_bin)};")
    if args.frame_index_json:
        manifest = json.dumps(json.loads(args.frame_index_json.read_text(encoding="utf-8")))
        statements.append(f"globalThis.__QT_H5_LOCAL_FRAME_INDEX={manifest};")
    if args.frame_low8 == "true":
        statements.append("globalThis.__BSLZ4_FRAME_LOW8=true;")
    elif args.frame_low8 == "false":
        statements.append("globalThis.__BSLZ4_FRAME_LOW8=false;")
    if args.u32_shared_low8:
        statements.append("globalThis.__BSLZ4_LOW8_U32_SHARED=true;")
    if args.single_parse_low8:
        statements.append("globalThis.__BSLZ4_SINGLE_PARSE_LOW8=true;")
    if args.frame_serial_low8:
        statements.append("globalThis.__BSLZ4_FRAME_SERIAL_LOW8=true;")
    if args.frames_per_wg is not None:
        statements.append(f"globalThis.__BSLZ4_FRAMES_PER_WG={int(args.frames_per_wg)};")
    if args.frame_wg is not None:
        statements.append(f"globalThis.__BSLZ4_FRAME_WG={int(args.frame_wg)};")
    if args.no_pipeline_staging:
        statements.append("globalThis.__BSLZ4_PIPELINE_STAGING=false;")
    if args.upload == "write-buffer":
        statements.append("globalThis.__BSLZ4_UPLOAD_WRITEBUFFER=true;")
        statements.append("globalThis.__BSLZ4_UPLOAD_MAPPED=false;")
        statements.append("globalThis.__BSLZ4_UPLOAD_COMBINED=false;")
    elif args.upload == "mapped":
        statements.append("globalThis.__BSLZ4_UPLOAD_WRITEBUFFER=false;")
        statements.append("globalThis.__BSLZ4_UPLOAD_MAPPED=true;")
        statements.append("globalThis.__BSLZ4_UPLOAD_COMBINED=false;")
    elif args.upload == "combined":
        statements.append("globalThis.__BSLZ4_UPLOAD_WRITEBUFFER=false;")
        statements.append("globalThis.__BSLZ4_UPLOAD_MAPPED=false;")
        statements.append("globalThis.__BSLZ4_UPLOAD_COMBINED=true;")
    elif args.upload == "staging":
        statements.append("globalThis.__BSLZ4_UPLOAD_WRITEBUFFER=false;")
        statements.append("globalThis.__BSLZ4_UPLOAD_MAPPED=false;")
        statements.append("globalThis.__BSLZ4_UPLOAD_COMBINED=false;")
    return "\n".join(statements)


def _parse_int_tuple(text: str, n: int, label: str) -> list[int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != n:
        raise ValueError(f"{label} must contain {n} comma-separated integers")
    return [int(part) for part in parts]


def _fixture_files(args: argparse.Namespace) -> list[str]:
    base = args.fixture_dir.rstrip("/")
    files = [f"{base}/{args.master_name}"]
    files.extend(
        f"{base}/{args.data_template.format(index=index)}"
        for index in range(1, args.data_files + 1)
    )
    if args.block_index_template:
        files.extend(
            f"{base}/{args.block_index_template.format(index=index)}"
            for index in range(1, args.data_files + 1)
        )
    missing = [file for file in files if not Path(file).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} missing total)"
        raise FileNotFoundError(f"Fixture file(s) not found: {preview}{suffix}")
    return files


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [int(run["profile"]["totalMs"]) for run in runs]
    walls = [int(run["wallMs"]) for run in runs]
    summary: dict[str, Any] = {
        "n": len(runs),
        "allParity": all(bool(run.get("parity", True)) for run in runs),
        "totalProfileMsSum": sum(totals),
        "totalProfileMsMedian": statistics.median(totals),
        "totalProfileMsMin": min(totals),
        "totalProfileMsMax": max(totals),
        "wallMsSum": sum(walls),
        "wallMsMedian": statistics.median(walls),
        "wallMsMin": min(walls),
        "wallMsMax": max(walls),
    }
    dpc = _dpc_summary(runs)
    if dpc:
        summary["dpc"] = dpc
    return summary


def _dpc_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    samples: dict[str, list[float]] = {}
    fps_values: list[float] = []
    all_lengths: list[int] = []
    out: dict[str, Any] = {}
    for run in runs:
        dpc = run.get("dpc") or {}
        raf = dpc.get("raf") or {}
        if isinstance(raf.get("fps"), (int, float)):
            fps_values.append(float(raf["fps"]))
        for sample in dpc.get("samples") or []:
            kind = sample.get("kind")
            source = sample.get("source")
            value = sample.get("value") or {}
            elapsed = sample.get("elapsedMs")
            if isinstance(value.get("length"), int):
                all_lengths.append(int(value["length"]))
            if isinstance(elapsed, (int, float)) and kind and source:
                samples.setdefault(f"{source}:{kind}", []).append(float(elapsed))
        for parity in dpc.get("parity") or []:
            source = parity.get("source")
            if not source:
                continue
            out.setdefault("parity", {})[source] = parity

    if not samples and not fps_values and "parity" not in out:
        return {}
    for key, values in sorted(samples.items()):
        out[key] = {
            "n": len(values),
            "medianMs": round(statistics.median(values), 3),
            "minMs": round(min(values), 3),
            "maxMs": round(max(values), 3),
        }
    if fps_values:
        out["raf"] = {
            "n": len(fps_values),
            "medianFps": round(statistics.median(fps_values), 2),
            "minFps": round(min(fps_values), 2),
        }
    if all_lengths:
        out["pixelLengths"] = sorted(set(all_lengths))
    return out


def _dpc_probe_script(reps: int, fps_ms: int, warmup: int, dpc_references: dict[str, str] | None) -> str:
    references_json = json.dumps(dpc_references or {})
    return f"""(async () => {{
      const api = globalThis.__sh4d || null;
      if (!api) return {{ available: false, error: "__sh4d missing" }};
      const sources = ["DPC_row", "DPC_col", "iDPC"];
      const samples = [];
      const parity = [];
      const references = {references_json};
      const measure = async (kind, source, fn) => {{
        const t0 = performance.now();
        const value = await fn();
        samples.push({{
          kind,
          source,
          elapsedMs: performance.now() - t0,
          value,
        }});
      }};
      const display = async (source) => {{
        if (typeof api.dpcDisplayOnly === "function") {{
          return await api.dpcDisplayOnly(source);
        }}
        if (typeof api.dpcBufferOnly === "function") {{
          return await api.dpcBufferOnly(source);
        }}
        return {{ available: false, error: "DPC display hook missing" }};
      }};
      for (const source of sources) {{
        for (let i = 0; i < {int(warmup)}; i++) {{
          await display(source);
          if (typeof api.dpcOnly === "function") await api.dpcOnly(source);
        }}
      }}
      for (const source of sources) {{
        for (let i = 0; i < {int(reps)}; i++) {{
          await measure("display", source, () => display(source));
        }}
      }}
      for (const source of sources) {{
        for (let i = 0; i < {int(reps)}; i++) {{
          if (typeof api.dpcOnly === "function") await measure("readback", source, () => api.dpcOnly(source));
        }}
      }}
      for (const source of sources) {{
        for (let i = 0; i < {int(reps)}; i++) {{
          if (api.model && typeof api.model.set === "function" && typeof api.recomputeVI === "function") {{
            await measure("recomputeVI", source, async () => {{
              api.model.set("vi_source", source);
              await api.recomputeVI();
              await new Promise((resolve) => requestAnimationFrame(resolve));
              return {{
                profile: globalThis.__sh4dViProfile || null,
                display: globalThis.__sh4dDpcDisplay || null,
              }};
            }});
          }}
        }}
      }}
      if (typeof api.dpcCompareReference === "function") {{
        for (const source of sources) {{
          if (references[source]) {{
            const t0 = performance.now();
            const got = await api.dpcCompareReference(source, references[source]);
            parity.push({{ ...got, elapsedMs: performance.now() - t0 }});
          }}
        }}
      }}
      const fpsWindowMs = Math.max(250, {int(fps_ms)});
      let frames = 0;
      const rafStart = performance.now();
      await new Promise((resolve) => {{
        const step = () => {{
          frames += 1;
          if (performance.now() - rafStart >= fpsWindowMs) resolve();
          else requestAnimationFrame(step);
        }};
        requestAnimationFrame(step);
      }});
      const rafElapsedMs = performance.now() - rafStart;
      return {{
        available: true,
        samples,
        parity,
        raf: {{
          frames,
          elapsedMs: rafElapsedMs,
          fps: frames * 1000 / Math.max(1, rafElapsedMs),
        }},
        profile: globalThis.__loadprof || null,
        softwareAdapter: Boolean(globalThis.__loadprof && globalThis.__loadprof.softwareAdapter),
        adapterInfo: globalThis.__loadprof ? globalThis.__loadprof.adapterInfo : null,
      }};
    }})()"""


def _run_one(
    args: argparse.Namespace,
    files: list[str],
    reference_checksums: list[dict[str, Any]] | None,
    dpc_references: dict[str, str] | None,
    index: int,
) -> dict[str, Any]:
    target = CdpTarget(args.cdp, "about:blank")
    try:
        target.call("Page.enable")
        target.call("Runtime.enable")
        target.call("DOM.enable")
        target.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1500, "height": 950, "deviceScaleFactor": 1, "mobile": False},
        )
        prelude = _runtime_prelude(args)
        if prelude:
            target.call("Page.addScriptToEvaluateOnNewDocument", {"source": prelude})
        target.call("Page.navigate", {"url": args.html_url})
        for _ in range(120):
            if target.eval("document.readyState") == "complete":
                break
            time.sleep(0.25)
        node = 0
        for _ in range(80):
            root = target.call("DOM.getDocument", {"depth": -1})["root"]["nodeId"]
            node = target.call("DOM.querySelector", {"nodeId": root, "selector": "input[type=file]"})["nodeId"]
            if node:
                break
            time.sleep(0.25)
        if not node:
            title = target.eval("document.title")
            body = target.eval("document.body ? document.body.innerText.slice(0, 500) : ''")
            raise RuntimeError(f"file input not found; title={title!r} body={body!r}")
        wall0 = time.perf_counter()
        target.call("DOM.setFileInputFiles", {"nodeId": node, "files": files}, timeout=60)
        mounted_count = target.eval(
            """(() => {
              const input = document.querySelector('input[type=file]');
              input.dispatchEvent(new Event('input', { bubbles: true }));
              input.dispatchEvent(new Event('change', { bubbles: true }));
              return input.files.length;
            })()"""
        )
        if mounted_count != len(files):
            raise RuntimeError(
                f"Browser mounted {mounted_count} file(s), expected {len(files)}"
            )
        state = None
        while time.perf_counter() - wall0 < args.timeout_s:
            state = target.eval(
                f"""(() => ({{
                  loadprof: globalThis.__loadprof || null,
                  hasChecksums: Boolean(globalThis.__sh4d && globalThis.__sh4d.rawChecksums)
                }}))()"""
            )
            profile = (state or {}).get("loadprof")
            local_ok = bool(profile) and (
                not args.require_local_profile or bool(profile.get("localFiles"))
            )
            if profile and profile.get("frames") == args.frames and state.get("hasChecksums") and local_ok:
                break
            if profile and local_ok and profile.get("frames") not in (None, args.frames):
                raise RuntimeError(
                    "browser load completed with an unexpected frame count: "
                    f"got {profile.get('frames')}, expected {args.frames}; "
                    "check the exported widget scan shape or use a shape-explicit harness"
                )
            time.sleep(0.25)
        else:
            raise TimeoutError(f"load did not finish; last state={state}")
        checksums = None
        parity = None
        if reference_checksums is not None:
            middle = max(0, args.frames // 2)
            last = max(0, args.frames - 1)
            checksums = target.eval(
                f"globalThis.__sh4d.rawChecksums([0,{middle},{last}])",
                timeout=30,
                await_promise=True,
            )
            parity = checksums == reference_checksums
        dpc = None
        if args.dpc_reps > 0:
            dpc = target.eval(
                _dpc_probe_script(args.dpc_reps, args.fps_ms, args.dpc_warmup, dpc_references),
                timeout=max(30, args.timeout_s),
                await_promise=True,
            )
        if args.screenshot and index == args.reps:
            png = target.call("Page.captureScreenshot", {"format": "png", "fromSurface": True}, timeout=30)["data"]
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            args.screenshot.write_bytes(base64.b64decode(png))
        return {
            "rep": index,
            "wallMs": round((time.perf_counter() - wall0) * 1000),
            "profile": state["loadprof"],
            "checksums": checksums,
            "parity": parity,
            "dpc": dpc,
        }
    finally:
        target.close(args.cdp)


def main() -> None:
    args = _parse_args()
    reference_checksums = None
    if args.checksum_json:
        reference_checksums = json.loads(args.checksum_json.read_text(encoding="utf-8"))["checksums"]
    dpc_references = None
    if args.dpc_reference_json:
        manifest = json.loads(args.dpc_reference_json.read_text(encoding="utf-8"))
        base_url = args.html_url.rsplit("/", 1)[0] + "/"
        dpc_references = {
            str(name): urllib.parse.urljoin(base_url, str(filename))
            for name, filename in (manifest.get("files") or {}).items()
        }
    files = _fixture_files(args)
    runs = []
    for index in range(1, args.reps + 1):
        run = _run_one(args, files, reference_checksums, dpc_references, index)
        runs.append(run)
        print(
            json.dumps(
                {
                    "rep": index,
                    "totalMs": run["profile"].get("totalMs"),
                    "wallMs": run["wallMs"],
                    "variant": run["profile"].get("decodeVariant"),
                    "workers": run["profile"].get("workerCount"),
                    "groupSize": run["profile"].get("groupSize"),
                    "decodeBatch": run["profile"].get("decodeBatch"),
                    "parity": run.get("parity"),
                }
            ),
            flush=True,
        )
    out = {
        "kind": "webgpu-h5-browser-benchmark",
        "html": "external-export",
        "fixtureFiles": len(files),
        "settings": {
            "workers": args.workers,
            "groupSize": args.group_size,
            "decodeBatch": args.decode_batch,
            "frameLow8": args.frame_low8,
            "u32SharedLow8": bool(args.u32_shared_low8),
            "singleParseLow8": bool(args.single_parse_low8),
            "frameSerialLow8": bool(args.frame_serial_low8),
            "framesPerWg": args.frames_per_wg,
            "frameWg": args.frame_wg,
            "pipelineStaging": not bool(args.no_pipeline_staging),
            "upload": args.upload,
            "sourceScanShape": args.source_scan_shape,
            "scanRegion": args.scan_region,
            "frameIndex": bool(args.frame_index_json),
            "blockIndex": bool(args.block_index_template),
            "dpcReps": args.dpc_reps,
            "dpcWarmup": args.dpc_warmup,
            "fpsMs": args.fps_ms,
        },
        "runs": runs,
        "summary": _summary(runs),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
