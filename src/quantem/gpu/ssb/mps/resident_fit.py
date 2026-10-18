"""Resident joint-model scores with extended-precision band reduction."""

from quantem.gpu.ssb._execution import checkpoint
from quantem.gpu.ssb._resident_search import _Device
from quantem.gpu.ssb.mps.thick_sample import (
    _FIT_DISPATCH_BATCH,
    _FIT_PENDING_BYTES,
    _K_CHUNK,
    _band,
    _thick_fit_batch_kernel,
    _trial_probe_kernel,
)

REDUCE = r"""
if(index>=BATCH*256)return;
int b=int(index)/256,lane=int(index)%256;Real total(0);
for(int q=lane;q<BAND;q+=256){
    Real den(sums[2*BATCH*BAND+b*BAND+q]);
    if(den>Real(0)){
        Real re(sums[b*BAND+q]),im(sums[BATCH*BAND+b*BAND+q]);
        total=total+Real(weight[q])*(re*re+im*im)/den;
    }
}
store_real(out0,int(index),total);
"""


def evaluate(prepared, trial, band_inv_A=(0.2, 0.9)):
    """Evaluate resident encoded trials without a score or parameter readback."""
    device = _Device("mps")
    mx = device.xp
    flat, band_qx, band_qy, weight = _band(prepared, band_inv_A)
    weight = mx.array(weight, dtype=mx.float32)
    n_band = int(flat.size)
    g = prepared.g_qk
    storage_bf = int(g.shape[0])
    plane = int(g.shape[1]) * int(g.shape[2])
    chunks = (storage_bf + _K_CHUNK - 1) // _K_CHUNK
    scalars = mx.array(
        [
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
            prepared.factor,
        ],
        dtype=mx.float32,
    )
    # Schedule independent tiles together, including the eight search trials.
    # Keep each thread's four-candidate arithmetic and BF accumulation order;
    # only the grid's independent tile dimension grows. Target the partial-sum
    # budget using the actual band size, with at least one four-candidate tile.
    per_candidate = (chunks + 1) * n_band * 3 * 4
    execution = max(
        _FIT_DISPATCH_BATCH,
        (_FIT_PENDING_BYTES // per_candidate // _FIT_DISPATCH_BATCH)
        * _FIT_DISPATCH_BATCH,
    )
    outputs = []
    for start in range(0, len(trial), execution):
        checkpoint()
        rows = trial[start : start + execution]
        batch = len(rows)
        tile = min(_FIT_DISPATCH_BATCH, batch)
        probe_geometry, probe_chi = _trial_probe_kernel()(
            inputs=[prepared.kx, prepared.ky, scalars, rows.reshape(-1)],
            template=[("NUM_BF", storage_bf), ("B", batch)],
            grid=(storage_bf, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(storage_bf, 4), (storage_bf, batch)],
            output_dtypes=[mx.float32] * 2,
        )
        parts = _thick_fit_batch_kernel()(
            inputs=[
                g,
                flat,
                band_qx,
                band_qy,
                prepared.kx,
                prepared.ky,
                rows.reshape(-1),
                scalars,
                probe_geometry,
                probe_chi,
            ],
            template=[
                ("B", tile),
                ("TOTAL", batch),
                ("N_BAND", n_band),
                ("NUM_BF", storage_bf),
                ("PLANE", plane),
                ("K_CHUNK", _K_CHUNK),
            ],
            grid=(n_band, chunks, (batch + tile - 1) // tile),
            threadgroup=(64, 1, 1),
            output_shapes=[(chunks, batch, n_band)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        sums = mx.stack([mx.sum(part, axis=0) for part in parts])
        constants = f"\n#define BATCH {batch}\n#define BAND {n_band}\n"
        partial = device.run(
            "ssb_joint_reduce_band",
            {"sums": sums, "weight": weight},
            [(batch, 256, 3)],
            REDUCE,
            count=batch * 256,
            constants=constants,
        )[0]
        score = device.run(
            "ssb_joint_reduce_score",
            {"partial": partial},
            [(batch, 3)],
            "if(index>=BATCH)return;Real total(0);for(int i=0;i<256;i++)total=total+load_real(partial,int(index)*256+i);store_real(out0,int(index),total);",
            count=batch,
            constants=constants,
        )[0]
        outputs.append(score)
    return mx.concatenate(outputs, axis=0)
