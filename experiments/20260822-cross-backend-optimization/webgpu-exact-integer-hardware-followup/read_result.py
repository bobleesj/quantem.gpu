"""Read the bounded exact-integer WebGPU harness result over Chrome CDP."""

from __future__ import annotations

import json
import sys
import time
import urllib.request

import websocket


def main() -> int:
    deadline = time.monotonic() + 60
    pages: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:9337/json/list", timeout=1) as response:
                pages = json.load(response)
        except Exception:
            time.sleep(0.2)
            continue
        target = next(
            (
                page
                for page in pages
                if page.get("type") == "page"
                and "qgpu-webgpu-exact-integer" in str(page.get("url", ""))
            ),
            None,
        )
        if target is None:
            time.sleep(0.2)
            continue
        ws = websocket.create_connection(str(target["webSocketDebuggerUrl"]), timeout=5)
        try:
            request_id = 1
            while time.monotonic() < deadline:
                ws.send(
                    json.dumps(
                        {
                            "id": request_id,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": "window.__qgpuDone ? JSON.stringify(window.__qgpuResult) : null",
                                "returnByValue": True,
                            },
                        }
                    )
                )
                while True:
                    message = json.loads(ws.recv())
                    if message.get("id") == request_id:
                        break
                value = message.get("result", {}).get("result", {}).get("value")
                if value:
                    result = json.loads(value)
                    print(json.dumps(result, indent=2, sort_keys=True))
                    return 0 if result.get("status") == "passed" else 2
                request_id += 1
                time.sleep(0.2)
        finally:
            ws.close()
    print(json.dumps({"status": "timeout", "pages": pages}, indent=2), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
