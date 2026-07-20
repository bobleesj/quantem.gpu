#!/usr/bin/env python
"""Benchmark WebGPU product-first HDF5 virtual-image decode in Chrome."""

from __future__ import annotations

import argparse
import base64
import json
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
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--master-name", default="sample_master.h5")
    parser.add_argument("--data-template", default="data_{index:06d}.h5")
    parser.add_argument(
        "--sidecar-template",
        help=(
            "Selected-block product sidecar template. When set, the harness "
            "selects these exact raw-count sidecars instead of native data files; "
            "the sidecars are the product evidence payload for the requested mask."
        ),
    )
    parser.add_argument("--data-files", type=int, default=27)
    parser.add_argument("--frame-index-json", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--masked-sum-wg", type=int, choices=(64, 128, 256))
    parser.add_argument("--groupmask", action="store_true")
    parser.add_argument(
        "--pixel-mask",
        action="store_true",
        help="Force the older pixel-index masked-sum kernel instead of the default group-mask kernel.",
    )
    parser.add_argument(
        "--no-masked-sum-pipeline",
        action="store_true",
        help="Disable the selected-block masked-sum staging pipeline.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--screenshot", type=Path)
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


def _prelude(args: argparse.Namespace) -> str:
    if args.groupmask and args.pixel_mask:
        raise ValueError("--groupmask and --pixel-mask are mutually exclusive")
    statements: list[str] = [
        "globalThis.__BSLZ4_LOW8_ONLY=true;",
        "globalThis.__BSLZ4_FRAME_LOW8=true;",
    ]
    if args.masked_sum_wg:
        statements.append(f"globalThis.__QT_BSLZ4_MASKED_SUM_WG={args.masked_sum_wg};")
    if args.groupmask:
        statements.append("globalThis.__QT_BSLZ4_MASKED_SUM_GROUPMASK=true;")
    elif args.pixel_mask:
        statements.append("globalThis.__QT_BSLZ4_MASKED_SUM_GROUPMASK=false;")
    if args.no_masked_sum_pipeline:
        statements.append("globalThis.__QT_BSLZ4_MASKED_SUM_PIPELINE=false;")
    if args.frame_index_json:
        manifest = json.dumps(json.loads(args.frame_index_json.read_text(encoding="utf-8")))
        statements.append(f"globalThis.__QT_H5_LOCAL_FRAME_INDEX={manifest};")
    return "\n".join(statements)


def _fixture_files(args: argparse.Namespace) -> list[str]:
    base = args.fixture_dir.rstrip("/")
    files = [f"{base}/{args.master_name}"]
    template = args.sidecar_template or args.data_template
    files.extend(f"{base}/{template.format(index=index)}" for index in range(1, args.data_files + 1))
    files.append(args.reference_file)
    return files


def _run(args: argparse.Namespace) -> dict[str, Any]:
    target = CdpTarget(args.cdp, "about:blank")
    try:
        target.call("Page.enable")
        target.call("Runtime.enable")
        target.call("DOM.enable")
        target.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1500, "height": 950, "deviceScaleFactor": 1, "mobile": False},
        )
        target.call("Page.addScriptToEvaluateOnNewDocument", {"source": _prelude(args)})
        url = args.html_url
        if args.batch:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}batch={args.batch}"
        target.call("Page.navigate", {"url": url})
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
            body = target.eval("document.body ? document.body.innerText.slice(0, 500) : ''")
            raise RuntimeError(f"file input not found; body={body!r}")
        files = _fixture_files(args)
        wall0 = time.perf_counter()
        target.call("DOM.setFileInputFiles", {"nodeId": node, "files": files}, timeout=60)
        selected_count = target.eval(
            """(() => {
              const input = document.querySelector('input[type=file]');
              return input.files.length;
            })()"""
        )
        started = target.eval(
            """(() => {
              const input = document.querySelector('input[type=file]');
              if (typeof globalThis.__runProductFirst !== 'function') {
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
                return {direct:false, selected: input.files.length, hasRunner:false};
              }
              globalThis.__runProductFirst(input.files).catch((err) => {
                globalThis.__productProf = {
                  error: err instanceof Error ? (err.stack || err.message) : String(err),
                  localH5: globalThis.__QT_LOCAL_H5_DEBUG || null
                };
              });
              return {direct:true, selected: input.files.length, hasRunner:true};
            })()"""
        )
        if int(selected_count or 0) == 0:
            raise RuntimeError(f"no files selected; start={started}")
        state = None
        while time.perf_counter() - wall0 < args.timeout_s:
            state = target.eval(
                """(() => ({
                  product: globalThis.__productProf || null,
                  localH5: globalThis.__QT_LOCAL_H5_DEBUG || null
                }))()"""
            )
            product_state = (state or {}).get("product")
            if product_state and (product_state.get("parity") or product_state.get("error")):
                state = product_state
                break
            time.sleep(0.25)
        else:
            raise TimeoutError(f"product-first benchmark did not finish; last={state}")
        if state.get("error"):
            raise RuntimeError(state["error"])
        if args.screenshot:
            png = target.call("Page.captureScreenshot", {"format": "png", "fromSurface": True}, timeout=30)["data"]
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            args.screenshot.write_bytes(base64.b64decode(png))
        return {"wallMs": round((time.perf_counter() - wall0) * 1000), "profile": state}
    finally:
        target.close(args.cdp)


def main() -> None:
    args = _parse_args()
    result = {
        "kind": "webgpu-h5-product-first-browser-benchmark",
        "html": "external-export",
        "settings": {
            "batch": args.batch,
            "frameIndex": bool(args.frame_index_json),
            "maskedSumWg": args.masked_sum_wg,
            "maskedSumMode": "groupmask" if args.groupmask else "pixel" if args.pixel_mask else "default",
            "maskedSumPipeline": not bool(args.no_masked_sum_pipeline),
        },
        "run": _run(args),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prof = result["run"]["profile"]
    local_profile = prof.get("profile", {}) if isinstance(prof, dict) else {}
    product_profile = local_profile.get("productProfile", {}) if isinstance(local_profile, dict) else {}
    print(json.dumps({
        "wallMs": result["run"]["wallMs"],
        "totalMs": prof.get("totalMs"),
        "sourceMode": local_profile.get("sourceMode"),
        "productMs": local_profile.get("productMs"),
        "variant": product_profile.get("variant"),
        "kernelTotalMs": product_profile.get("totalMs"),
        "gpuWaitMs": product_profile.get("gpuWaitMs"),
        "uploadMs": product_profile.get("uploadMs"),
        "selectedCompressedMB": product_profile.get("compressedMB"),
        "selectedGroups": product_profile.get("selectedGroups"),
        "maxAbs": prof.get("parity", {}).get("maxAbs"),
        "meanAbs": prof.get("parity", {}).get("meanAbs"),
        "mismatch": prof.get("parity", {}).get("mismatch"),
    }, indent=2))


if __name__ == "__main__":
    main()
