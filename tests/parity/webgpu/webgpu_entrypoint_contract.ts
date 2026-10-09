/** Import/lifecycle compatibility only; this is not a hardware benchmark. */
import * as api from "../../../src/quantem/gpu/webgpu/index";
import * as dense from "../../../src/quantem/gpu/io/hdf5/webgpu/local-h5";
import { DetectorCompute } from "../../../src/quantem/gpu/detector/webgpu/backend";

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

check(api.loadLocalH5Master === dense.loadLocalH5Master, "Dense loader was duplicated");
check(api.loadLocalH5MaskedSum === dense.loadShow4DSTEMLocalH5MaskedSum, "Dense sum was duplicated");
check(api.DetectorCompute === DetectorCompute, "Detector implementation changed");
check(!("GPUColormapEngine" in api), "Browser display belongs to quantem.widget, not quantem.gpu");

// The renamed entry point must use the existing file registry, not a second one.
api.setLocalFiles([{ name: "example_master.h5" } as File]);
check(dense.show4DSTEMHasLocalFiles(), "Legacy and canonical registries diverged");
api.clearLocalFiles();
check(!dense.show4DSTEMHasLocalFiles(), "Canonical cleanup left a legacy source reference");
console.log("WebGPU entry-point identities and file-registry cleanup passed");
