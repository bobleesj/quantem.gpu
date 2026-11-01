"""Run diffraction-to-iDPC parity on an existing headed WebGPU browser.

Bundle dpc-python-parity.ts as DpcParity into fixtures/test.js first. Fixtures
contain manifest.json, uint16 detector counts, uint32 masks and float32 Python
reference outputs. Supply --fixtures and --cdp; no browser is silently launched.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--cdp', required=True)
    args = parser.parse_args()
    root = args.fixtures.resolve()
    with sync_playwright() as browser_api:
        browser = browser_api.chromium.connect_over_cdp(args.cdp)
        page = browser.contexts[0].new_page()
        def serve(route):
            name = urlparse(route.request.url).path.lstrip('/')
            if name == '':
                route.fulfill(body='<html><body>DPC hardware parity</body></html>', content_type='text/html')
                return
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                route.abort()
                return
            route.fulfill(body=path.read_bytes(), content_type='application/octet-stream')
        page.route('https://dpc-parity.test/**', serve)
        page.goto('https://dpc-parity.test/')
        page.add_script_tag(content=(root / 'test.js').read_text())
        result = page.evaluate('''async () => {
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter || adapter.isFallbackAdapter) throw new Error('Hardware WebGPU adapter required');
          const device = await adapter.requestDevice();
          device.pushErrorScope('validation');
          const tests = await DpcParity.run(device, 'https://dpc-parity.test/');
          const error = await device.popErrorScope();
          if (error) throw new Error(error.message);
          return {adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description},tests};
        }''')
        (root / 'browser-results.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        page.close()


if __name__ == '__main__':
    main()
