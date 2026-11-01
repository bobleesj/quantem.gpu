/// <reference types="@webgpu/types" />
/** Compare the actual detector-to-iDPC GPU path with independent Python fixtures. */
import { DetectorCompute } from '../../src/quantem/gpu/detector/webgpu/backend';
export async function run(device: GPUDevice, base: string) {
  const cases = await (await fetch(base + 'manifest.json')).json();
  const results = [];
  for (const item of cases) {
    const raw = await (await fetch(base + item.raw)).arrayBuffer();
    const mask = new Uint32Array(await (await fetch(base + item.mask)).arrayBuffer());
    const buffer = device.createBuffer({size:raw.byteLength,usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(raw));buffer.unmap();
    const compute = DetectorCompute.fromGpuChunks(device,[{buffer,startScan:0,nScan:item.rows*item.cols}],item.rows*item.cols,item.detRows*item.detCols,0);
    try {
      for (const test of item.tests) {
        const expected = new Float32Array(await (await fetch(base + test.expected)).arrayBuffer());
        const actual = await compute.maskedIDpc(mask,item.detCols,item.rows,item.cols,test.angle,test.flipped);
        let squared=0, reference=0, maxError=0;
        for(let i=0;i<actual.length;i++) {
          const error=actual[i]-expected[i];squared+=error*error;reference+=expected[i]*expected[i];maxError=Math.max(maxError,Math.abs(error));
        }
        const relativeRms=Math.sqrt(squared/Math.max(reference,1e-20));
        if(!Number.isFinite(relativeRms) || relativeRms>2e-4) throw new Error(`${item.name} ${test.angle} ${test.flipped}: relative RMS ${relativeRms}`);
        results.push({name:item.name,angle:test.angle,flipped:test.flipped,relativeRms,maxError});
      }
    } finally {compute.dispose();}
  }
  return results;
}
