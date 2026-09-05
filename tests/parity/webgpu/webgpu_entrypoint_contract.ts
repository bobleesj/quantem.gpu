/** Import/lifecycle compatibility only; this is not a hardware benchmark. */
import * as api from "../../../src/quantem/gpu/webgpu/index";
import * as dense from "../../../src/quantem/gpu/io/backends/webgpu/local-h5";
import * as packed from "../../../src/quantem/gpu/io/backends/webgpu/compact-h5";
import { DetectorCompute } from "../../../src/quantem/gpu/detector/backends/webgpu/backend";
import { GPUColormapEngine } from "../../../src/quantem/gpu/display/backends/webgpu/colormaps";

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

check(api.loadLocalH5Master === dense.loadLocalH5Master, "Dense loader was duplicated");
check(api.loadLocalH5MaskedSum === dense.loadShow4DSTEMLocalH5MaskedSum, "Dense sum was duplicated");
check(api.loadCompactH5WebGPU === packed.loadCompactH5WebGPU, "Packed loader was duplicated");
check(api.parseCompactH5Index === packed.parseCompactH5Index, "Packed parser was duplicated");
check(api.WebGPUCompactH5ResidentSource === packed.WebGPUCompactH5ResidentSource, "Resident type changed");
check(api.CompactH5SessionQualificationCache === packed.CompactH5SessionQualificationCache, "Qualification cache changed");
check(api.DetectorCompute === DetectorCompute, "Detector implementation changed");
check(api.GPUColormapEngine === GPUColormapEngine, "Display implementation changed");

// The renamed entry point must use the existing file registry, not a second one.
api.setLocalFiles([{ name: "example_master.h5" } as File]);
check(dense.show4DSTEMHasLocalFiles(), "Legacy and canonical registries diverged");
api.clearLocalFiles();
check(!dense.show4DSTEMHasLocalFiles(), "Canonical cleanup left a legacy source reference");
console.log("WebGPU entry-point identities and file-registry cleanup passed");
