"""Device-resident multivariate TPE and trust-region Newton joint fitting.

Only shape/scheduling metadata crosses to the host during optimization. Trial
points, model scores, density ranking, finite differences, eigensolves and
trust-region decisions remain on the selected GPU. Results are downloaded
once for the public fit record. The search retains groups of eight, 24 TPE
candidates, the startup budget, and three distinct Newton starts.

Objective: ``fit_resident`` (tilt=True) maximizes the least-squares
agreement of the thick-sample SSB model with G(q, k) on CUDA and MPS.
``fit_standard_resident`` (tilt=False) is the same search and objective with
the sample tilt and depth spread held at 0, on CUDA and MPS
(``cuda/optimizer.py``, ``mps/optimizer.py``). The phase-variance search
(``objective="phase_variance"``) is ``mps/optimizer.optimize_phase_variance``
on MPS and ``phase_variance_fit.py`` on CUDA.
"""

import math
import re
import time
from functools import lru_cache

import numpy as np

from quantem.gpu.direct_ptycho.thick_sample_fit import DEPTH_SPREAD_RANGE_A
from quantem.gpu.direct_ptycho._execution import checkpoint
from quantem.gpu.direct_ptycho._resident_precision import HEADER


class _Device:
    """Compile the same decision kernels for Metal and CUDA."""

    def __init__(self, backend: str):
        self.backend = backend
        if backend == "mps":
            from quantem.gpu.direct_ptycho.mps.hardware import require_mlx

            self.xp = require_mlx()
        elif backend == "cuda":
            import cupy

            self.xp = cupy
        else:
            raise ValueError(
                f"Resident joint fitting requires mps or cuda, got {backend!r}."
            )

    def pack(self, values):
        # Public scalar configuration crosses once, losslessly, before fitting.
        values = np.asarray(values, dtype=np.float64)
        high = values.astype(np.float32)
        middle = (values - high).astype(np.float32)
        low = (values - high - middle).astype(np.float32)
        return self.xp.array(np.stack((high, middle, low), axis=-1))

    def host(self, values):
        self.sync(values)
        raw = np.asarray(values) if self.backend == "mps" else values.get()
        return raw.astype(np.float64).sum(axis=-1)

    def sync(self, *values):
        if self.backend == "mps":
            self.xp.eval(*values)
        else:
            self.xp.cuda.get_current_stream().synchronize()

    def run(self, name, inputs, shapes, body, *, count=1, constants=""):
        # Counts and dispatch offsets must not compile a new shader for each
        # adaptive batch or Newton iteration. They are launch metadata only.
        launch = []

        def parameterize(match):
            key, value = match.groups()
            if key not in {
                "COUNT",
                "DONE",
                "BATCH",
                "OFFSET",
                "ACTIVE",
                "ITERATION",
                "TOTAL",
                "BAND",
            }:
                return match.group(0)
            slot = len(launch)
            launch.append(int(value))
            return f"#define {key} int(launch_constants[{slot}])"

        constants = re.sub(
            r"#define ([A-Z_]+) ([0-9]+)(?![0-9a-z])", parameterize, constants
        )
        if launch:
            inputs = {
                **inputs,
                "launch_constants": self.xp.array(launch, dtype=self.xp.float32),
            }
        names = tuple(inputs)
        result = _kernel(self.backend, name, names, len(shapes), body, constants)
        if self.backend == "mps":
            return result(
                inputs=list(inputs.values()),
                grid=(count, 1, 1),
                threadgroup=(min(128, count), 1, 1),
                output_shapes=shapes,
                output_dtypes=[self.xp.float32] * len(shapes),
            )
        outputs = [self.xp.empty(shape, dtype=self.xp.float32) for shape in shapes]
        result(((count + 127) // 128,), (128,), (*inputs.values(), *outputs))
        return outputs


@lru_cache(maxsize=512)
def _kernel(backend, name, names, n_outputs, body, constants):
    outputs = [f"out{i}" for i in range(n_outputs)]
    if backend == "mps":
        from quantem.gpu.direct_ptycho.mps.hardware import require_mlx

        return require_mlx().fast.metal_kernel(
            name=name,
            input_names=list(names),
            output_names=outputs,
            header=HEADER + constants,
            source="uint index=thread_position_in_grid.x;\n" + body,
            compile_options={"math_mode": "safe"},
        )
    import cupy as cp

    signature = ",".join(
        [f"const float *{n}" for n in names] + [f"float *{n}" for n in outputs]
    )
    source = (
        HEADER + constants + f'\nextern "C" __global__ void {name}({signature}){{\n'
    )
    source += "unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;\n" + body + "\n}"
    # A bounded six-dimensional solve contains nested fixed-size loops. Letting
    # NVRTC inline and unroll every trust-region solve makes cold compilation
    # grow by minutes without providing useful parallelism in this one thread.
    for function in ("trust_step", "bounded_step"):
        source = source.replace(
            f"FN void {function}", f"__device__ __noinline__ void {function}"
        )
    source = re.sub(r"\bfor\s*\(", "\n#pragma unroll 1\nfor(", source)
    return cp.RawKernel(source, name, options=("--std=c++17", "--fmad=false"))


# MT19937 and the 53-bit uniform construction used by the existing sampler.
# Startup and TPE use separate streams, both initialized with the fit seed.
RANDOM_HEADER = r"""
FN unsigned int next_mt(LOCAL unsigned int *state, LOCAL int &position) {
    if(position==624){
        for(int i=0;i<624;i++){
            unsigned int y=(state[i]&0x80000000u)|(state[(i+1)%624]&0x7fffffffu);
            state[i]=state[(i+397)%624]^(y>>1)^((y&1u)?0x9908b0dfu:0u);
        }
        position=0;
    }
    unsigned int y=state[position++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;return y;
}
"""

RANDOM_BODY = r"""
if(index>0)return;
unsigned int state[624];state[0]=SEED;
for(int i=1;i<624;i++)state[i]=1812433253u*(state[i-1]^(state[i-1]>>30))+i;
int position=624;
for(int i=0;i<TOTAL;i++){
    unsigned int a=next_mt(state,position)>>5,b=next_mt(state,position)>>6;
    out0[4*i]=float(a>>16);out0[4*i+1]=float(a&65535);
    out0[4*i+2]=float(b>>16);out0[4*i+3]=float(b&65535);
}
"""

UNIFORM_BODY = r"""
if(index>=TOTAL)return;
int i=int(index);
Real a=Real(words[4*i])*Real(65536)+Real(words[4*i+1]);
Real b=Real(words[4*i+2])*Real(65536)+Real(words[4*i+3]);
store_real(out0,i,(a*Real(67108864.0f)+b)/Real(9007199254740992.0f));
"""

DENSITY_BODY = r"""
if(index>=DONE+1)return;
int i=int(index), good_count=min(25,(DONE+9)/10), rank=0, group_order=0;
if(i<DONE)rank=int(ranks[i]);
bool good=rank<good_count;
if(i==DONE)good=false;
if(i<DONE)for(int j=0;j<i;j++)if((int(ranks[j])<good_count)==good)group_order++;
int n=good?good_count:DONE-good_count;
Real sigma=max_r((Real(1)/Real(5))*exp_r(-log_r(Real(max(n,1)))/Real(DIMENSIONS+4)),
                 Real(1)/Real(min(100,n+2)));
if(i==DONE)sigma=Real(1);
Real norm=Real(0);
for(int d=0;d<6;d++){
    if((FIXED && d<3)||(THIN && d>=3))continue;
    Real mu=i<DONE?load_real(history,i*6+d):Real(0.5f);
    Real mass=cdf_r((Real(1)-mu)/sigma)-cdf_r(-mu/sigma);
    norm=norm+log_r(mass)+log_r(sigma);
}
Real weight(1);
if(n>=25 && group_order<n-25){
    weight=n==26?Real(1)/Real(n):Real(1)/Real(n)+(Real(1)-Real(1)/Real(n))*Real(group_order)/Real(n-26);
}
store_real(out0,i*5,good?Real(1):Real(0));
store_real(out0,i*5+1,sigma);store_real(out0,i*5+2,norm);
store_real(out0,i*5+3,weight);store_real(out0,i*5+4,Real(group_order));
"""

SAMPLE_BODY = r"""
if(index>=BATCH*24*6)return;
int d=int(index)%6,candidate=int(index)/6,b=candidate/24,c=candidate%24;
if((FIXED && d<3)||(THIN && d>=3)){store_real(out0,int(index),Real(0));return;}
int below=min(25,(DONE+9)/10);
Real choose=load_real(random,(OFFSET+b)*(DIMENSIONS+1)*24+c)*Real(below+1);
int chosen=int(scalar(choose)),selected=DONE;
for(int i=0;i<DONE;i++)if(load_real(density,i*5)>Real(0.5f) && int(scalar(load_real(density,i*5+4)))==chosen)selected=i;
Real sigma=selected==DONE?Real(1):load_real(density,selected*5+1);
Real mu=selected==DONE?Real(0.5f):load_real(history,selected*6+d);
// Optuna's multivariate dimensions are alphabetical after startup.
int order=d<3?d:(d==3?5:(d==4?4:3));
if(FIXED)order-=3;
Real q=load_real(random,(OFFSET+b)*(DIMENSIONS+1)*24+24+order*24+c);
Real lo=cdf_r(-mu/sigma),hi=cdf_r((Real(1)-mu)/sigma);
Real p=clip_r(lo+q*(hi-lo),Real(1e-18f),Real(1)-Real(1e-18f));
store_real(out0,int(index),clip_r(mu+sigma*inverse_cdf_r(p),Real(0),Real(1)));
"""

SELECT_BODY = r"""
if(index>=BATCH)return;
int b=int(index),best=0;
for(int c=1;c<24;c++)if(load_real(scores,b*24+c)>load_real(scores,b*24+best))best=c;
for(int d=0;d<6;d++)store_real(out0,b*6+d,load_real(candidates,(b*24+best)*6+d));
"""

DECODE_BODY = r"""
if(index>=COUNT*6)return;
int d=int(index)%6;
Real lo=load_real(bounds,d),hi=load_real(bounds,6+d);
store_real(out0,int(index),lo+load_real(points,int(index))*(hi-lo));
"""

TRIAL_BODY = r"""
if(index>=COUNT)return;
int b=int(index);Real angle=load_real(points,b*6+2)*Real(2);
out0[b*7]=scalar(load_real(points,b*6));out0[b*7+1]=scalar(load_real(points,b*6+1));
out0[b*7+2]=scalar(cos_r(angle));out0[b*7+3]=scalar(sin_r(angle));
out0[b*7+4]=scalar(load_real(points,b*6+5));
out0[b*7+5]=scalar(load_real(points,b*6+3)/Real(1000));
out0[b*7+6]=scalar(load_real(points,b*6+4)/Real(1000));
"""


def _random(device, seed, trials, width):
    constants = (
        RANDOM_HEADER
        + f"\n#define SEED {seed & 0xFFFFFFFF}u\n#define TOTAL {trials * width}\n"
    )
    # MT state transitions are serial. The expensive extended-precision
    # conversion is independent for every number, so give it a GPU thread.
    # Four exactly represented 16-bit words retain the original 53-bit stream.
    words = device.run(
        "ssb_joint_random_words",
        {},
        [(trials * width, 4)],
        RANDOM_BODY,
        constants=constants,
    )[0]
    return device.run(
        "ssb_joint_uniform",
        {"words": words},
        [(trials, width, 3)],
        UNIFORM_BODY,
        count=trials * width,
        constants=f"\n#define TOTAL {trials * width}\n",
    )[0]


def _decode(device, points, bounds):
    return device.run(
        "ssb_joint_decode",
        {"points": points, "bounds": bounds},
        [points.shape],
        DECODE_BODY,
        count=points.shape[0] * 6,
        constants=f"\n#define COUNT {points.shape[0]}\n",
    )[0]


def _trial(device, points):
    return device.run(
        "ssb_joint_trial",
        {"points": points},
        [(points.shape[0], 7)],
        TRIAL_BODY,
        count=points.shape[0],
        constants=f"\n#define COUNT {points.shape[0]}\n",
    )[0]


RANK_BODY = r"""
if(index>=COUNT)return;
int i=int(index),rank=0;Real value=load_real(scores,i);
for(int j=0;j<COUNT;j++){
    Real other=load_real(scores,j);
    if(other>value || (!(other<value)&&j<i))rank++;
}
out0[i]=float(rank);
"""

def _rank(device, scores):
    return device.run(
        "ssb_joint_rank",
        {"scores": scores},
        [(len(scores),)],
        RANK_BODY,
        count=len(scores),
        constants=f"\n#define COUNT {len(scores)}\n",
    )[0]


def _search(device, evaluate, bounds, trials, seed, *, fixed=False, thin=False):
    xp = device.xp
    startup = max(30, trials // 6) if not thin else 10
    dimensions = 3 if (fixed or thin) else 6
    flags = f"\n#define FIXED {int(fixed)}\n#define THIN {int(thin)}\n#define DIMENSIONS {dimensions}\n"
    uniform = _random(device, seed, trials, dimensions)
    tpe_random = _random(device, seed, trials, (dimensions + 1) * 24)
    history = xp.zeros((0, 6, 3), dtype=xp.float32)
    scores = xp.zeros((0, 3), dtype=xp.float32)
    tpe_offset = 0
    for done in range(0, trials, 8):
        checkpoint()
        batch = min(8, trials - done)
        constants = (
            flags
            + f"\n#define DONE {done}\n#define BATCH {batch}\n#define OFFSET {tpe_offset}\n"
        )
        if done < startup:
            points = uniform[done : done + batch]
            if dimensions == 3:
                zero = xp.zeros((batch, 3, 3), dtype=xp.float32)
                points = xp.concatenate(
                    (zero, points) if fixed else (points, zero), axis=1
                )
        else:
            ranks = _rank(device, scores)
            density = device.run(
                "ssb_joint_density",
                {"history": history, "scores": scores, "ranks": ranks},
                [(done + 1, 5, 3)],
                DENSITY_BODY,
                count=done + 1,
                constants=constants,
            )[0]
            candidates = device.run(
                "ssb_joint_sample",
                {"history": history, "density": density, "random": tpe_random},
                [(batch, 24, 6, 3)],
                SAMPLE_BODY,
                count=batch * 24 * 6,
                constants=constants,
            )[0]
            components = device.run(
                "ssb_joint_components",
                {"history": history, "density": density, "candidates": candidates},
                [(batch, 24, done + 1, 3)],
                COMPONENT_BODY,
                count=batch * 24 * (done + 1),
                constants=constants,
            )[0]
            maxima = device.run(
                "ssb_joint_mixture_max",
                {"density": density, "components": components},
                [(batch, 24, 4, 3)],
                MAX_MIXTURE_BODY,
                count=batch * 24,
                constants=constants,
            )[0]
            powers = device.run(
                "ssb_joint_mixture_exp",
                {"density": density, "components": components, "maxima": maxima},
                [(batch, 24, done + 1, 2, 3)],
                EXP_MIXTURE_BODY,
                count=batch * 24 * (done + 1),
                constants=constants,
            )[0]
            ratios = device.run(
                "ssb_joint_mixture_sum",
                {"components": powers, "maxima": maxima},
                [(batch, 24, 3)],
                SUM_MIXTURE_BODY,
                count=batch * 24,
                constants=constants,
            )[0]
            points = device.run(
                "ssb_joint_select",
                {"candidates": candidates, "scores": ratios},
                [(batch, 6, 3)],
                SELECT_BODY,
                count=batch,
                constants=constants,
            )[0]
            tpe_offset += batch
        real_points = _decode(device, points, bounds)
        values = evaluate(_trial(device, real_points))
        history = xp.concatenate((history, points), axis=0)
        scores = xp.concatenate((scores, values), axis=0)
        device.sync(history, scores)
    return _decode(device, history, bounds), scores


# Polish once more from starts in other basins (``_exact_newton.other_basins``) and keep the higher agreement.
BASIN_CHECK = True


def fit_resident(
    evaluate,
    *,
    backend: str,
    trials: int = 300,
    tilt_limit_mrad: float = 25.0,
    depth_spread_range_A: tuple[float, float] = DEPTH_SPREAD_RANGE_A,
    c10_range: tuple[float, float] = (-300.0, 300.0),
    c12_max: float = 200.0,
    seed: int = 0,
    verbose: bool = True,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    polish_starts: int = 3,
    polish: str = "newton",
    compare_standard: bool = True,
    start=None,
    fixed_aberrations: dict[str, float] | None = None,
    standard_only: bool = False,
    derivatives=None,
) -> dict[str, object]:
    """Fit the joint model with resident adaptive search and Newton refinement.

    Parameters use the existing engine units. ``evaluate`` accepts resident
    encoded model trials and returns resident extended scores. Public callers
    use ``SSB.find_aberrations(tilt=True)``; no solver setting is needed.
    ``derivatives`` (``LowOrderDerivatives`` of ``cuda`` or ``mps.exact_derivatives``) are the
    exact gradient and Hessian of the same objective that the trust-region
    Newton (``_exact_newton``) climbs; a fit that polishes needs them.
    ``fixed_aberrations`` holds C10, C12 and phi12 at those values.
    ``standard_only=True`` runs only the standard comparison (the 200-trial
    search at depth spread 0 that ``compare_standard`` adds) and returns its
    ``standard_fit``, ``standard`` and timing: the same values a full fit with
    the same seed and bounds reports.
    """
    from quantem.gpu.direct_ptycho.thick_sample_fit import PARAMETERS

    if polish not in ("newton", None):
        raise ValueError("Resident joint fitting supports refinement='newton' or None.")
    if trials < 0:
        raise ValueError(
            "trials must be non-negative; use 200 for a complete joint search."
        )
    if trials == 0 and start is None and not standard_only:
        raise ValueError("Zero-trial refinement needs a previous joint fit as start.")
    if standard_only and not compare_standard:
        raise ValueError("standard_only runs the standard comparison: leave compare_standard=True.")
    if not 0 <= polish_starts <= 3:
        raise ValueError("Resident Newton supports zero to three distinct starts.")
    if polish is not None and polish_starts and derivatives is None:
        raise ValueError("The resident Newton polishes on the backend's exact derivatives: pass derivatives.")
    device = _Device(backend)
    lower = [
        c10_range[0],
        0.0,
        -math.pi / 2,
        -tilt_limit_mrad,
        -tilt_limit_mrad,
        depth_spread_range_A[0],
    ]
    upper = [
        c10_range[1],
        c12_max,
        math.pi / 2,
        tilt_limit_mrad,
        tilt_limit_mrad,
        depth_spread_range_A[1],
    ]
    if fixed_aberrations is not None:
        for d, name in enumerate(PARAMETERS[:3]):
            lower[d] = upper[d] = fixed_aberrations[name]
    bounds = device.pack(lower + upper)
    timings = {}
    begin = time.perf_counter()
    standard = None
    standard_fit = None
    if compare_standard:
        thin_bounds = device.pack(
            lower[:3] + [0.0, 0.0, 0.0] + upper[:3] + [0.0, 0.0, 0.0]
        )
        thin_points, thin_scores = _search(
            device, evaluate, thin_bounds, 200, seed, thin=True
        )
        thin_ranks = _rank(device, thin_scores)
        thin_point, thin_value = device.run(
            "ssb_joint_best",
            {"points": thin_points, "scores": thin_scores, "ranks": thin_ranks},
            [(1, 6, 3), (1, 3)],
            "if(index>0)return;for(int i=0;i<COUNT;i++)if(int(ranks[i])==0){for(int d=0;d<6;d++)store_real(out0,d,load_real(points,i*6+d));store_real(out1,0,load_real(scores,i));}",
            constants="\n#define COUNT 200\n",
        )
        standard = dict(
            zip(PARAMETERS[:3], device.host(thin_point)[0, :3].tolist(), strict=True)
        )
        standard_fit = float(device.host(thin_value)[0])
    timings["standard_comparison_s"] = time.perf_counter() - begin
    if standard_only:
        return {"standard_fit": standard_fit, "standard": standard, "timings": timings}
    begin = time.perf_counter()
    if trials:
        points, scores = _search(
            device,
            evaluate,
            bounds,
            int(trials),
            int(seed),
            fixed=fixed_aberrations is not None,
        )
    else:
        points = device.pack([start])
        scores = evaluate(_trial(device, points))
        device.sync(scores)
    timings["coarse_search_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    if polish is None:
        polish_starts = 0
    exact = polish_starts > 0
    if exact:
        from quantem.gpu.direct_ptycho import _exact_newton

        host_point, fit, exact_starts = _exact_newton.fit(
            derivatives, device.host(points), device.host(scores), lower, upper, start=start,
            starts=polish_starts, basins=bool(trials) and fixed_aberrations is None and BASIN_CHECK)
    else:
        ranks = _rank(device, scores)
        point, value = device.run(
            "ssb_joint_best",
            {"points": points, "scores": scores, "ranks": ranks},
            [(1, 6, 3), (1, 3)],
            "if(index>0)return;for(int i=0;i<COUNT;i++)if(int(ranks[i])==0){for(int d=0;d<6;d++)store_real(out0,d,load_real(points,i*6+d));store_real(out1,0,load_real(scores,i));}",
            constants=f"\n#define COUNT {len(points)}\n",
        )
    if not exact:
        device.sync(point, value)
        host_point = device.host(point)[0]
        fit = float(device.host(value)[0])
    timings["refinement_s"] = time.perf_counter() - begin
    best = dict(zip(PARAMETERS, np.asarray(host_point, dtype=float).tolist(), strict=True))
    if not math.isfinite(fit) or fit <= 0:
        raise ValueError(
            "Joint tilt fitting found no positive finite agreement. Check scan sampling and signal in the fitting band, or use tilt=False, objective='phase_variance'."
        )
    record_points = device.host(points)
    record_scores = device.host(scores)
    records = (
        [
            {
                "number": i,
                "params": dict(zip(PARAMETERS, row.tolist(), strict=True)),
                "loss": -float(score),
            }
            for i, (row, score) in enumerate(
                zip(record_points, record_scores, strict=True)
            )
        ]
        if trials
        else []
    )
    best["phi12"] = (best["phi12"] + math.pi / 2) % math.pi - math.pi / 2
    diagnostics = exact_starts if exact else []
    return {
        **best,
        "fit": fit,
        "standard_fit": standard_fit,
        "gain": None
        if standard_fit is None
        else fit / standard_fit
        if standard_fit > 0
        else float("nan"),
        "standard": standard,
        "band_inv_A": list(band_inv_A),
        "trials": int(trials),
        "trial_records": records,
        "timings": timings,
        "optimizer": {
            "search": "resident multivariate TPE",
            "refinement": polish if polish_starts else None,
            "derivatives": "exact" if exact else None,
            "device": backend,
            "host_numerical_search": False,
            "precision": "three-float expansion" if backend == "mps" else "float64",
            "newton_starts": diagnostics,
        },
    }


def fit_standard_resident(
    evaluate,
    *,
    backend: str,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    trials: int = 200,
    seed: int = 42,
    polish: str | None = "newton",
    lock: tuple[int, ...] = (),
    start=None,
    polish_starts: int = 3,
    derivatives=None,
) -> dict[str, object]:
    """Standard SSB fit (tilt=False): the joint fit's search and objective with sample tilt and depth spread at 0.

    ``lower`` and ``upper`` bound (C10, C12, phi12) in engine units (Angstrom, Angstrom, rad); a coefficient with
    ``lower == upper`` is held at that value in every trial. The search is the resident multivariate TPE of the
    joint fit's standard comparison (``_search(thin=True)``, 10 startup trials). ``polish``:

    - ``"newton"``: the trust-region Newton on the exact gradient and Hessian (``derivatives``, the backend's
      ``LowOrderDerivatives``; ``_exact_newton`` with the tilt and depth spread held at 0) from the best
      ``polish_starts`` distinct trials (and ``start``), on (C10, C12 cos 2 phi12, C12 sin 2 phi12). It keeps C10 in
      its range and C12 below its maximum, so it needs a C12 range from 0 and the full phi12 period, and no ``lock``;
    - ``"nelder-mead"``: bounded SciPy Nelder-Mead from the same starts, the starts' simplices evaluated together.
      ``lock`` (indices into (C10, C12, phi12)) holds those coefficients at the best trial's values; phi12 wraps
      when its range is the whole period;
    - ``None``: the best trial (or ``start``) as is.

    ``trials=0`` polishes from ``start`` alone. Returns the fitted point, its agreement and the trial records.
    """
    from quantem.gpu.direct_ptycho.batch_refinement import _minimize_batch

    if trials < 0:
        raise ValueError("trials must be non-negative; use 200 for a complete standard search.")
    if trials == 0 and start is None:
        raise ValueError("Zero-trial refinement needs a start point.")
    if polish not in ("newton", "nelder-mead", None):
        raise ValueError("A standard fit polishes with 'newton', 'nelder-mead' or None.")
    full_period = lower[2] <= -math.pi / 2 + 1e-9 and upper[2] >= math.pi / 2 - 1e-9
    if polish == "newton" and (lock or lower[1] != 0.0 or not full_period):
        raise ValueError("Newton refinement bounds C10 and the C12 maximum only; use 'nelder-mead' to lock a "
                         "coefficient, bound phi12 or hold a nonzero C12.")
    if polish == "newton" and polish_starts and derivatives is None:
        raise ValueError("The standard Newton polishes on the backend's exact derivatives: pass derivatives.")
    device = _Device(backend)
    bounds = device.pack(list(lower) + [0.0, 0.0, 0.0] + list(upper) + [0.0, 0.0, 0.0])
    timings = {}
    begin = time.perf_counter()
    if trials:
        points, scores = _search(device, evaluate, bounds, int(trials), int(seed), thin=True)
    else:
        points = device.pack([[*start[:3], 0.0, 0.0, 0.0]])
        scores = evaluate(_trial(device, points))
        device.sync(scores)
    timings["coarse_search_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    host_points = device.host(points)[:, :3]
    host_scores = device.host(scores)
    order = np.argsort(-host_scores, kind="stable")
    nfev = 0
    if polish == "newton" and polish_starts:
        from quantem.gpu.direct_ptycho import _exact_newton

        point, fit, starts = _exact_newton.fit(
            derivatives, device.host(points), host_scores, [*lower, 0.0, 0.0, 0.0], [*upper, 0.0, 0.0, 0.0],
            start=None if start is None or not trials else [*start[:3], 0.0, 0.0, 0.0], starts=polish_starts,
            basins=False)
        best = point[:3]
        nfev = int(sum(entry["evaluations"] for entry in starts))
    elif polish == "nelder-mead" and polish_starts:
        candidates = ([np.asarray(start[:3], dtype=np.float64)] if start is not None and trials else []) + [
            host_points[index] for index in order]
        starts: list[np.ndarray] = []
        for candidate in candidates:
            if len(starts) >= polish_starts:
                break
            if all(np.max(np.abs(candidate - kept) / np.array([50.0, 20.0, 0.3])) > 1.0 for kept in starts):
                starts.append(np.array(candidate, dtype=np.float64))
        held = host_points[order[0]] if trials else np.asarray(start[:3], dtype=np.float64)
        free = [index for index in range(3) if index not in lock]
        for kept in starts:
            kept[list(lock)] = held[list(lock)]
        scale = max(abs(float(host_scores[order[0]])), 1e-30)
        if not free:
            best, fit = starts[0], float(host_scores[order[0]])
        else:
            def rows(free_points):
                full = np.tile(starts[0], (len(free_points), 1))
                full[:, free] = free_points
                full[:, 1] = np.abs(full[:, 1])
                if full_period:
                    full[:, 2] = (full[:, 2] + math.pi / 2) % math.pi - math.pi / 2
                return full

            def objective(free_points):
                full = rows(np.asarray(free_points, dtype=np.float64))
                packed = device.pack(np.concatenate((full, np.zeros((len(full), 3))), axis=1))
                return -device.host(evaluate(_trial(device, packed))) / scale

            box = [(lower[0], upper[0]), (lower[1], upper[1]), (None, None) if full_period else (lower[2], upper[2])]
            polished = _minimize_batch(objective, [kept[free] for kept in starts], bounds=[box[i] for i in free],
                                       options={"xatol": 0.05, "fatol": 1e-6, "maxiter": 400})
            outcome = min(polished, key=lambda result: result.fun)
            best, fit = rows(outcome.x[None])[0], float(-outcome.fun) * scale
            nfev = int(sum(result.nfev for result in polished))
    else:
        best, fit = host_points[order[0]], float(host_scores[order[0]])
    timings["refinement_s"] = time.perf_counter() - begin
    if not math.isfinite(fit) or fit <= 0:
        raise ValueError("The standard SSB fit found no positive finite model agreement. Check the scan sampling and "
                         "the signal in the fitting band.")
    best = np.asarray(best, dtype=np.float64)
    best[1] = abs(best[1])
    best[2] = (best[2] + math.pi / 2) % math.pi - math.pi / 2
    records = [{"number": i, "params": {"C10": float(row[0]), "C12": float(row[1]), "phi12": float(row[2])},
                "loss": -float(score)} for i, (row, score) in enumerate(zip(host_points, host_scores, strict=True))
               ] if trials else []
    return {"C10": float(best[0]), "C12": float(best[1]), "phi12": float(best[2]), "fit": fit, "trials": int(trials),
            "trial_records": records, "timings": timings, "nfev": nfev,
            "optimizer": {"search": "resident multivariate TPE", "refinement": polish if polish_starts else None,
                          "device": backend, "objective": "agreement, depth spread 0"}}

# Probability components are independent work. Expose them as GPU threads
# instead of serially evaluating every mixture component within a candidate.
COMPONENT_BODY = r"""
if(index>=BATCH*24*(DONE+1))return;
int component=int(index)%(DONE+1),candidate=int(index)/(DONE+1);
Real width=load_real(density,component*5+1),sum(0);
for(int d=0;d<6;d++){
    if((FIXED&&d<3)||(THIN&&d>=3))continue;
    Real mu=component==DONE?Real(0.5f):load_real(history,component*6+d);
    Real delta=load_real(candidates,candidate*6+d)-mu;sum=sum+delta*delta;
}
Real weight=component==DONE?Real(1):load_real(density,component*5+3);
store_real(out0,int(index),-sum/(Real(2)*width*width)-load_real(density,component*5+2)+log_r(weight));
"""
MAX_MIXTURE_BODY = r"""
if(index>=BATCH*24)return;
int candidate=int(index);Real good(-1e30f),bad(-1e30f),good_w(0),bad_w(0);
for(int i=0;i<=DONE;i++){
    bool prior=i==DONE,below=load_real(density,i*5)>Real(0.5f);
    Real value=load_real(components,candidate*(DONE+1)+i),w=prior?Real(1):load_real(density,i*5+3);
    if(prior||below){good=max_r(good,value);good_w=good_w+w;}
    if(prior||!below){bad=max_r(bad,value);bad_w=bad_w+w;}
}
store_real(out0,candidate*4,good);store_real(out0,candidate*4+1,bad);
store_real(out0,candidate*4+2,good_w);store_real(out0,candidate*4+3,bad_w);
"""
EXP_MIXTURE_BODY = r"""
if(index>=BATCH*24*(DONE+1))return;
int component=int(index)%(DONE+1),candidate=int(index)/(DONE+1);
bool prior=component==DONE,below=load_real(density,component*5)>Real(0.5f);
Real value=load_real(components,int(index));
store_real(out0,int(index)*2,(prior||below)?exp_r(value-load_real(maxima,candidate*4)):Real(0));
store_real(out0,int(index)*2+1,(prior||!below)?exp_r(value-load_real(maxima,candidate*4+1)):Real(0));
"""
SUM_MIXTURE_BODY = r"""
if(index>=BATCH*24)return;
int candidate=int(index);Real good(0),bad(0);
for(int i=0;i<=DONE;i++){
    good=good+load_real(components,(candidate*(DONE+1)+i)*2);
    bad=bad+load_real(components,(candidate*(DONE+1)+i)*2+1);
}
store_real(out0,candidate,log_r(good)+load_real(maxima,candidate*4)-log_r(load_real(maxima,candidate*4+2))
                        -log_r(bad)-load_real(maxima,candidate*4+1)+log_r(load_real(maxima,candidate*4+3)));
"""
