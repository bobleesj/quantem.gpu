"""Run QEM and retained rANS product parity on a headed physical WebGPU adapter."""

import argparse
import functools
import http.server
import json
from pathlib import Path
import subprocess
import tempfile
import threading

from playwright.sync_api import sync_playwright

from make_qem_browser_fixture import generate


def main() -> None:
    """Build disposable fixtures and compare browser GPU products exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbuild", required=True, type=Path)
    parser.add_argument("--chrome", required=True, type=Path)
    args = parser.parse_args()
    sources = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="qem-browser-parity-") as temporary:
        root = Path(temporary)
        generate(root / "fixtures")
        for filename, exported in (
            ("count-ans-browser", "CountANSBrowserParity"),
            ("rans-products", "RansProductParity"),
            ("count-ans-series", "CountANSSeriesParity"),
            ("rans-integer-readback", "RansIntegerReadbackParity"),
        ):
            subprocess.run(
                [str(args.esbuild), str(sources / f"{filename}.ts"), "--bundle",
                 "--format=iife", f"--global-name={exported}",
                 f"--outfile={root / (filename + '.js')}"],
                check=True,
            )
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(http.server.SimpleHTTPRequestHandler, directory=root),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=str(args.chrome), headless=False,
                    args=["--enable-unsafe-webgpu"],
                )
                try:
                    page = browser.new_page()
                    page.goto(f"http://127.0.0.1:{server.server_port}")
                    page.add_script_tag(url="/count-ans-browser.js")
                    page.add_script_tag(url="/rans-products.js")
                    page.add_script_tag(url="/count-ans-series.js")
                    page.add_script_tag(url="/rans-integer-readback.js")
                    result = page.evaluate("""async () => {
                      const adapter = await navigator.gpu.requestAdapter();
                      if (!adapter || adapter.info.isFallbackAdapter) {
                        throw new Error('A physical WebGPU adapter is required');
                      }
                      const device = await adapter.requestDevice();
                      try {
                        return {
                          adapter: {
                            vendor: adapter.info.vendor,
                            architecture: adapter.info.architecture,
                            device: adapter.info.device,
                            description: adapter.info.description,
                            isFallbackAdapter: adapter.info.isFallbackAdapter,
                          },
                          qem: await CountANSBrowserParity.runCountANSBrowserParity(
                            device, location.origin + '/fixtures/'),
                          retained_rans: await RansProductParity.runRansProductParity(device),
                          ordered_qem_series: await CountANSSeriesParity.runCountANSSeriesParity(
                            device, location.origin + '/fixtures/'),
                          integer_readback: await RansIntegerReadbackParity.runRansIntegerReadbackParity(
                            device, location.origin + '/fixtures/saturated.qem'),
                        };
                      } finally { device.destroy(); }
                    }""")
                    print(json.dumps(result, indent=2))
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()
