"""Small synchronous Chrome DevTools Protocol transport for WebGPU runners."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

import websocket


def _http_json(cdp: str, method: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{cdp}{path}", method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


class CdpTarget:
    """Own one browser target and evaluate synchronous CDP requests."""

    def __init__(self, cdp: str, url: str) -> None:
        target = _http_json(
            cdp,
            "PUT",
            "/json/new?" + urllib.parse.quote(url, safe=""),
        )
        self.target_id = str(target["id"])
        self._ws = websocket.create_connection(
            target["webSocketDebuggerUrl"], timeout=20
        )
        self._next_id = 0

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> dict[str, Any]:
        """Call one CDP method, honoring its complete wall-time timeout."""
        self._next_id += 1
        msg_id = self._next_id
        self._ws.send(
            json.dumps(
                {"id": msg_id, "method": method, "params": params or {}}
            )
        )
        deadline = time.time() + timeout
        previous_timeout = self._ws.gettimeout()
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(method)
                self._ws.settimeout(remaining)
                try:
                    message = json.loads(self._ws.recv())
                except websocket.WebSocketTimeoutException as error:
                    raise TimeoutError(method) from error
                if message.get("id") != msg_id:
                    continue
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
        finally:
            self._ws.settimeout(previous_timeout)

    def eval(
        self,
        expression: str,
        *,
        timeout: float = 30,
        await_promise: bool = False,
    ) -> Any:
        """Evaluate JavaScript and return its by-value result."""
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

    def close(self) -> None:
        """Close the target if possible, then always close the socket."""
        try:
            self.call(
                "Target.closeTarget", {"targetId": self.target_id}, timeout=5
            )
        except (
            OSError,
            RuntimeError,
            TimeoutError,
            websocket.WebSocketException,
        ):
            pass
        finally:
            self._ws.close()
