"""Run the browser GPU parity gates (QEM, rANS products, dense HDF5) on a headed physical WebGPU adapter."""

import argparse
import functools
import http.server
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from urllib.parse import parse_qs, unquote, urlsplit

from playwright.sync_api import sync_playwright

from make_qem_browser_fixture import generate, generate_h5, generate_ssb, generate_ssb_higher_order


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the same bounded byte ranges that an exported SSB folder uses."""

    def send_head(self):
        self.remaining = None
        target = urlsplit(self.path)
        if unquote(target.path).endswith("/ssb columns.qem") and parse_qs(target.query).get("admission") != ["stream-check"]:
            self.send_error(403, "SSB fixture query token is required")
            return None
        value = self.headers.get("Range")
        if value is None:
            return super().send_head()
        path = Path(self.translate_path(self.path))
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", value)
        if not path.is_file() or match is None:
            self.send_error(416, "Requested range unavailable")
            return None
        first, last = map(int, match.groups())
        size = path.stat().st_size
        if not 0 <= first <= last < size:
            self.send_error(416, "Requested range outside file")
            return None
        source = path.open("rb")
        source.seek(first)
        self.remaining = last - first + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        while self.remaining:
            part = source.read(min(self.remaining, 1 << 20))
            if not part:
                break
            outputfile.write(part)
            self.remaining -= len(part)


def main() -> None:
    """Build disposable fixtures and compare browser GPU products exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esbuild", required=True, type=Path)
    parser.add_argument("--chrome", required=True, type=Path)
    parser.add_argument(
        "--expect-adapter", metavar="VENDOR/ARCHITECTURE",
        help="fail unless WebGPU reports this adapter, e.g. nvidia/blackwell",
    )
    parser.add_argument(
        "--chrome-arg", action="append", default=[],
        help="extra Chrome flag, e.g. --enable-features=Vulkan; repeat as needed",
    )
    args = parser.parse_args()
    sources = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="qem-browser-parity-") as temporary:
        root = Path(temporary)
        generate(root / "fixtures")
        generate_h5(root / "fixtures")
        generate_ssb(root / "fixtures")
        generate_ssb_higher_order(root / "fixtures" / "ssb-higher-order")
        for filename, exported in (
            ("qem-browser", "QemBrowserParity"),
            ("rans-products", "RansProductParity"),
            ("qem-series", "QemSeriesParity"),
            ("rans-integer-readback", "RansIntegerReadbackParity"),
            ("reduce-frames-parity", "ReduceFramesParity"),
            ("dense-h5-parity", "DenseH5Parity"),
            ("ssb-bf-subset", "SsbBfSubset"),
            ("ssb-qem", "SsbQem"),
            ("ssb-interaction", "SsbInteraction"),
            ("ssb-higher-order-sign", "SsbHigherOrderSign"),
        ):
            subprocess.run(
                [str(args.esbuild), str(sources / f"{filename}.ts"), "--bundle",
                 "--format=iife", f"--global-name={exported}",
                 f"--outfile={root / (filename + '.js')}"],
                check=True,
            )
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(RangeHandler, directory=root),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=str(args.chrome), headless=False,
                    args=["--enable-unsafe-webgpu", *args.chrome_arg],
                )
                try:
                    page = browser.new_page()
                    page.goto(f"http://127.0.0.1:{server.server_port}")
                    page.add_script_tag(url="/qem-browser.js")
                    page.add_script_tag(url="/rans-products.js")
                    page.add_script_tag(url="/qem-series.js")
                    page.add_script_tag(url="/rans-integer-readback.js")
                    page.add_script_tag(url="/reduce-frames-parity.js")
                    page.add_script_tag(url="/dense-h5-parity.js")
                    page.add_script_tag(url="/ssb-bf-subset.js")
                    page.add_script_tag(url="/ssb-qem.js")
                    page.add_script_tag(url="/ssb-interaction.js")
                    page.add_script_tag(url="/ssb-higher-order-sign.js")
                    result = page.evaluate("""async (expected) => {
                      const adapter = await navigator.gpu.requestAdapter();
                      // SwiftShader, Chrome's CPU renderer, need not report itself as a fallback adapter
                      if (!adapter || adapter.info.isFallbackAdapter || adapter.info.architecture === 'swiftshader'
                          || adapter.info.vendor === 'google') {
                        throw new Error('A physical WebGPU adapter is required');
                      }
                      const found = adapter.info.vendor + '/' + adapter.info.architecture;
                      if (expected && found !== expected) {
                        throw new Error(`Expected the ${expected} adapter, found ${found}`);
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
                          qem_checksum_workers: await QemBrowserParity.runQemChecksumParity(),
                          qem: await QemBrowserParity.runQemBrowserParity(
                            device, location.origin + '/fixtures/'),
                          retained_rans: await RansProductParity.runRansProductParity(device, location.origin + '/fixtures/'),
                          ordered_qem_series: await QemSeriesParity.runQemSeriesParity(
                            device, location.origin + '/fixtures/'),
                          integer_readback: await RansIntegerReadbackParity.runRansIntegerReadbackParity(
                            device, location.origin + '/fixtures/saturated.qem'),
                          reduce_frames: await ReduceFramesParity.runReduceFramesParity(device),
                          dense_decode: await DenseH5Parity.runDenseDecodeParity(location.origin + '/fixtures/'),
                          product_first: await DenseH5Parity.runProductFirstParity(location.origin + '/fixtures/'),
                          resident: await DenseH5Parity.runResidentParity(location.origin + '/fixtures/'),
                          virtual_image: await DenseH5Parity.runDenseVirtualImageParity(device),
                          ssb_interaction: await SsbInteraction.runSsbInteractionParity(location.origin + "/fixtures/"),
                          ssb_qem: await SsbQem.runSsbQemParity(location.origin + '/fixtures/'),
                          ssb_bf_subset: await SsbBfSubset.runSsbBfSubsetParity(),
                          ssb_build_peak: await SsbBfSubset.runSsbBuildPeak(),
                          ssb_higher_order_sign: await SsbHigherOrderSign.runSsbHigherOrderSign(
                            location.origin + '/fixtures/ssb-higher-order/'),
                        };
                      } finally { device.destroy(); }
                    }""", args.expect_adapter)
                    page.route("**/checksum-blocked.html", lambda route: route.fulfill(
                        status=200,
                        headers={"Content-Type": "text/html", "Content-Security-Policy": "worker-src 'none'"},
                        body="<title>QEM checksum fallback</title>",
                    ))
                    page.goto(f"http://127.0.0.1:{server.server_port}/checksum-blocked.html")
                    page.add_script_tag(url="/qem-browser.js")
                    result["qem_checksum_csp_fallback"] = page.evaluate(
                        "QemBrowserParity.runQemChecksumParity(false)"
                    )
                    print(json.dumps(result, indent=2))
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()
