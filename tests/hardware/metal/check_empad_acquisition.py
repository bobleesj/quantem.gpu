"""Validate a real XML/RAW or EMD acquisition against independent file bytes.

The full source is packed on Metal. Ordered, repeated and boundary diffraction
selections must match bit for bit; BF/ABF/ADF/total cover every scan position.
Run with INPUT OUTPUT_DIRECTORY and EMPAD_SOURCE_PARITY_EXE configured.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def main():
    source, out = map(Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=False)
    if source.suffix.lower() in ('.h5', '.emd', '.hdf5'):
        with h5py.File(source, 'r') as f:
            ds = f['datacube_root/datacube/data']
            rows, cols, _, _ = ds.shape
            offset = ds.id.get_offset()
        pixels = np.memmap(source, mode='r', dtype='<f4', offset=offset,
                           shape=(rows * cols, 128, 128))
    else:
        xml = ET.parse(source).getroot()
        if xml.findtext('sensor/type') == 'EMPAD2':
            rows, cols = map(int, xml.findtext('scan/shape').strip('()').split(','))
            raw = source.parent / xml.findtext('rawfile/filename')
            pixels = np.memmap(raw, mode='r', dtype='<f4', shape=(rows * cols, 128, 128))
        else:
            scan = xml.find("scan_parameters[@mode='acquire']")
            rows, cols = int(scan.findtext('scan_resolution_y')), int(scan.findtext('scan_resolution_x'))
            raw = source.parent / xml.find('raw_file').attrib['filename']
            pixels = np.memmap(raw, mode='r', dtype='<f4', shape=(rows * cols, 130, 128))[:, :128]
    count = rows * cols
    indices = [count - 1, 0, cols - 1, cols, count // 2, 0, count // 3]
    dest = out / 'selected.bin'
    proc = subprocess.run([os.environ['EMPAD_SOURCE_PARITY_EXE'], str(source), str(dest),
                           ','.join(map(str, indices))], capture_output=True, text=True,
                          env={**os.environ, 'EMPAD_TEST_METAL': '1',
                               'EMPAD_TEST_BUDGET': str(12 * 1024**3), 'QGPU_EMPAD_LOAD_PROFILE': '1'},
                          timeout=600)
    (out / 'console.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    measured = np.fromfile(dest, dtype='<u4').reshape(len(indices), 128, 128)
    np.testing.assert_array_equal(measured, pixels[indices].view('<u4'))
    products = np.fromfile(str(dest) + '.products', dtype='<f4').reshape(4, count)
    rr, cc = np.indices((128, 128))
    d2 = (rr - 64)**2 + (cc - 64)**2
    masks = [d2 <= 256, (d2 >= 64) & (d2 <= 256),
             (d2 >= 1024) & (d2 <= 3969), np.ones((128, 128), bool)]
    evidence = {'source': str(source), 'shape': [rows, cols, 128, 128],
                'dtype': 'float32', 'selected_indices': indices, 'selected_bits_equal': True,
                'products': {}, 'scope': 'full packed resident; sampled DP bits; full-scan products'}
    for kind, (name, mask) in enumerate(zip(('BF', 'ABF', 'ADF', 'total'), masks)):
        expected = np.empty(count, np.float64)
        for start in range(0, count, 64):
            expected[start:start+64] = pixels[start:start+64, mask].sum(axis=1, dtype=np.float64)
        np.testing.assert_allclose(products[kind], expected, rtol=1e-6, atol=1e-6)
        evidence['products'][name] = {'max_absolute_error': float(np.nanmax(np.abs(products[kind]-expected)))}
    evidence['metadata'] = json.loads(Path(str(dest)+'.metadata.json').read_text())
    evidence['pass'] = True
    (out / 'result.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
