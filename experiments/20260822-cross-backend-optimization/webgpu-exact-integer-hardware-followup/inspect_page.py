"""Inspect the bounded harness without changing browser state."""

from __future__ import annotations

import json
import urllib.request

import websocket


pages = json.load(urllib.request.urlopen("http://127.0.0.1:9337/json/list"))
target = next(page for page in pages if page.get("type") == "page")
ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=5)
try:
    expression = "JSON.stringify({ready:document.readyState,done:window.__qgpuDone||false,result:window.__qgpuResult||null,text:document.body.innerText})"
    ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expression, "returnByValue": True}}))
    while True:
        message = json.loads(ws.recv())
        if message.get("id") == 1:
            print(message["result"]["result"].get("value"))
            break
finally:
    ws.close()
