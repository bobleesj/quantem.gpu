#!/usr/bin/env python
"""Benchmark a browser-local Show4DSTEM HDF5 load through Chrome CDP.

This is a maintainer harness for exported QuantEM widget HTML. It drives the
real browser File API, waits for the WebGPU local-HDF5 load profile, optionally
checks corrected-frame checksums, and writes a JSON artifact without recording
private raw-data paths.
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
    parser.add_argument("--data-files", type=int, default=27)
    parser.add_argument("--frames", type=int, default=262144)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--decode-batch", type=int)
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
    parser.add_argument("--checksum-json", type=Path)
    parser.add_argument(
        "--frame-index-json",
        type=Path,
        help="Optional metadata-only frame-offset manifest injected as __QT_H5_LOCAL_FRAME_INDEX.",
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
    if args.workers is not None:
        statements.append(f"globalThis.__QT_H5_LOCAL_WORKERS={int(args.workers)};")
    if args.group_size is not None:
        statements.append(f"globalThis.__QT_H5_LOCAL_GROUP={int(args.group_size)};")
    if args.decode_batch is not None:
        statements.append(f"globalThis.__QT_H5_DECODE_BATCH={int(args.decode_batch)};")
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


def _fixture_files(args: argparse.Namespace) -> list[str]:
    base = args.fixture_dir.rstrip("/")
    files = [f"{base}/{args.master_name}"]
    files.extend(
        f"{base}/{args.data_template.format(index=index)}"
        for index in range(1, args.data_files + 1)
    )
    return files


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [int(run["profile"]["totalMs"]) for run in runs]
    walls = [int(run["wallMs"]) for run in runs]
    return {
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


def _run_one(args: argparse.Namespace, files: list[str], reference_checksums: list[dict[str, Any]] | None, index: int) -> dict[str, Any]:
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
        target.eval(
            """(() => {
              const input = document.querySelector('input[type=file]');
              input.dispatchEvent(new Event('input', { bubbles: true }));
              input.dispatchEvent(new Event('change', { bubbles: true }));
              return input.files.length;
            })()"""
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
            if profile and profile.get("frames") == args.frames and state.get("hasChecksums"):
                break
            time.sleep(0.25)
        else:
            raise TimeoutError(f"load did not finish; last state={state}")
        checksums = None
        parity = None
        if reference_checksums is not None:
            checksums = target.eval("globalThis.__sh4d.rawChecksums([0,131072,262143])", timeout=30, await_promise=True)
            parity = checksums == reference_checksums
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
        }
    finally:
        target.close(args.cdp)


def main() -> None:
    args = _parse_args()
    reference_checksums = None
    if args.checksum_json:
        reference_checksums = json.loads(args.checksum_json.read_text(encoding="utf-8"))["checksums"]
    files = _fixture_files(args)
    runs = []
    for index in range(1, args.reps + 1):
        run = _run_one(args, files, reference_checksums, index)
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
            "frameIndex": bool(args.frame_index_json),
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
