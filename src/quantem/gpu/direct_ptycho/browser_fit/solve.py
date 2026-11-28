import json
import math
import time
import numpy as np
import optuna
from pyodide.ffi import run_sync
from policy._gradient_refinement import Settings, refine
from policy.fifth_order_parameters import aperture_phase, aperture_rms
from policy.higher_order import ORDERS

def fit(agreement, progress, wavelength, semiangle):
    def arguments(z, ring):
        higher = {}
        i = 6
        for n, m in ORDERS[2:]:
            scale = (n+1)*wavelength/(2*math.pi*semiangle**(n+1))
            x, y = z[i], z[i+1] if m else 0
            higher[f"C{n}{m}_A"] = scale*(math.hypot(x, y) if m else x)
            if m:
                higher[f"phi{n}{m}_deg"] = math.degrees(math.atan2(y, x))/m
            i += 1 + bool(m)
        options = dict(higherOrder=higher, sample=dict(sampleTiltMrad=(z[3:5]*5).tolist(), depthSpreadA=z[5]*100))
        if ring is not None:
            options["ring"] = ring
        return [100*z[0], 100*math.hypot(*z[1:3]), .5*math.atan2(z[2], z[1]), options]

    seed_gradients = {}

    def score(z, ring=None, gradient=False):
        if z.tobytes() in seed_gradients:
            entry = seed_gradients[z.tobytes()][0 if ring is None else ring + 1]
            return entry if gradient else entry[0]
        return json.loads(run_sync(agreement(json.dumps(arguments(z, ring)), gradient)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(
        seed=42, multivariate=True, n_startup_trials=33, n_ei_candidates=24))
    best = -math.inf
    started = time.perf_counter()
    for batch in range(25):
        trials = []
        for _ in range(8):
            t = study.ask()
            a,b,f,r,c,d = [t.suggest_float(name,lo,hi) for name,lo,hi in (
                ("C10",-300,300),("C12",0,200),("phi12",-math.pi/2,math.pi/2),
                ("tilt_row",-25,25),("tilt_col",-25,25),("thickness",20,600))]
            trials.append((t,np.array([a/100,b*math.cos(2*f)/100,b*math.sin(2*f)/100,r/5,c/5,d/100]+[0.]*22)))
        points = [arguments(z, None) for _, z in trials]
        scores = json.loads(run_sync(agreement(json.dumps(points), "batch")))
        for (t, z), value in zip(trials, scores, strict=True):
            study.tell(t,value)
            if value > best:
                best, start = value, z
        progress(f"Fitting · {(batch+1)*8}/200 trials")
    trials_s = time.perf_counter()-started
    bounds = np.array([(-3.,3.),(-2,2),(-2,2),(-5,5),(-5,5),(.2,6)]+[(-20,20)]*22)
    results=[]
    for ring in (0,1,None):
        progress("Fifth-order refinement · "+("full band" if ring is None else f"ring {ring+1}/2"))
        if ring is None:
            phase = aperture_phase(results,wavelength,semiangle)
            rms = float(aperture_rms(phase[:1],phase[1:])[0])
            if rms > .5:
                bounds[6:] = [-2,2]
        norm=score(start,ring)
        def evaluate(points,kind,stride):
            if ring == 0 and len(points) > 1 and kind == "value_and_gradient":
                for i in range(0, len(points), 4):
                    chunk = points[i:i+4]
                    args = [arguments(z, None) for z in chunk]
                    rows = json.loads(run_sync(agreement(json.dumps(args), "seed")))
                    for z, row in zip(chunk, rows, strict=True):
                        seed_gradients[z.tobytes()] = row
            values=[score(z,ring,kind!="value") for z in points]
            if kind=="value":
                return np.array(values)/norm,None
            return np.array([v[0] for v in values])/norm,np.array([v[1] for v in values])/norm
        result=refine(evaluate,start,bounds,settings=Settings(hessian_bf_stride=1,coarse_bf_stride=1))
        results.append(result["coordinates"])
    return json.dumps(dict(args=arguments(results[-1],None),trials_s=trials_s, refinement_s=time.perf_counter()-started-trials_s, split_rms=rms, gain=result["agreement"]))
