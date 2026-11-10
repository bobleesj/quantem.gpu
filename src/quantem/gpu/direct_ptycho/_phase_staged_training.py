"""Fit the three-stage residual architecture from one measured acquisition."""

import math
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import nn

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.direct_ptycho import direct_workflow
from quantem.gpu.direct_ptycho._annular_detector import annular_counts, annular_limits
from quantem.gpu.direct_ptycho._phase_networks import Direction, ResidualUNet
from quantem.gpu.direct_ptycho._phase_refinement import (
    _context,
    _direction_input,
    _readout,
    _scatter,
    readout_kernels,
)
from quantem.gpu.direct_ptycho._phase_training import _regions
from quantem.gpu.direct_ptycho.cuda.direct import PreparedDirect


@dataclass
class _Block:
    rows: torch.Tensor
    cols: torch.Tensor
    gram: torch.Tensor
    rhs: torch.Tensor
    prior: torch.Tensor
    response: torch.Tensor
    guide_response: torch.Tensor


class _Case:
    """Fixed equations and differentiable correction maps for one detector split."""

    def __init__(self, source, target, guide, sampling):
        self.source, self.target = source, target
        wave, _ = source.solve(0.03, None, compute_loss=False)
        self.base = torch.from_dlpack(cp.angle(wave)).detach()
        self.guide = guide
        self.area = source.factor**2
        self.sigma = self.base.std().clamp_min(1e-10)
        self.input = torch.stack((self.base / self.sigma, torch.ones_like(self.base)))[None]
        self.scale = torch.as_tensor(source.scale, device=self.base.device).float()
        self.power = torch.from_dlpack(target.signal_power)
        if not bool(torch.isfinite(self.power) & (self.power > 0)):
            raise ValueError(
                "No finite diffraction variation in the held detector block; select a patch with signal."
            )
        self.qr, self.qc = [
            torch.fft.fftfreq(n, s, device=self.base.device)
            for n, s in zip(self.base.shape, sampling)
        ]
        self.radius = torch.hypot(self.qr[:, None], self.qc[None, :])
        self.base_ft = torch.fft.fft2(self.base) / self.area
        self.blocks = []
        self.targets = []
        for a, b in zip(source.groups, target.groups, strict=True):
            r, c = [torch.from_dlpack(x) for x in (a.rows, a.cols)]
            gram, rhs, prior = [torch.from_dlpack(x) for x in (a.gram, a.rhs, a.prior)]
            gram, rhs, prior = gram.cfloat(), rhs.cfloat(), prior.float()
            d = 0.03 * self.scale * prior
            matrix = gram + torch.diag_embed(2 * d)
            low = 0.03 * self.scale * (0.15 / self.radius[r, c].clamp_min(1e-6)) ** 4
            response = torch.linalg.solve(matrix, torch.diag_embed(d).cfloat())
            guide_response = torch.linalg.solve(
                matrix + torch.diag_embed(low), torch.diag_embed(d).cfloat()
            )
            self.blocks.append(_Block(r, c, gram, rhs, prior, response, guide_response))
            self.targets.append(tuple(torch.from_dlpack(x).cfloat() for x in (b.gram, b.rhs)))

    def image(self, coefficients):
        spectrum = torch.zeros_like(self.base_ft)
        for b, value in zip(self.blocks, coefficients, strict=True):
            _scatter(spectrum, b.rows, b.cols, value)
        return torch.fft.ifft2(spectrum).real * self.area

    def correction(self, spectrum, *, guide=False):
        return [
            (b.guide_response if guide else b.response) @ spectrum[b.rows, b.cols, None]
            for b in self.blocks
        ]

    def unet_phase(self, network):
        proposal = torch.fft.fft2(self.sigma * network(self.input)[0, 0]) / self.area
        correction = self.correction(proposal)
        return self.base + self.image([x[..., 0] for x in correction])

    def gain(self, phase, reference=None):
        reference = self.base if reference is None else reference
        x = torch.fft.fft2(reference) / self.area
        delta = torch.fft.fft2(phase - reference) / self.area
        change = torch.zeros((), device=phase.device, dtype=torch.float64)
        for b, (gram, rhs) in zip(self.blocks, self.targets, strict=True):
            a, d = x[b.rows, b.cols], delta[b.rows, b.cols]
            gradient = (gram @ a[..., None])[..., 0] - rhs
            change += (
                2 * (d.conj() * gradient).real + (d.conj() * (gram @ d[..., None])[..., 0]).real
            ).sum(dtype=torch.float64)
        return -change / self.power

    @torch.no_grad()
    def readout_features(self, kernels):
        features = []
        for index in range(18):
            weights = torch.zeros(18, device=self.base.device)
            weights[index] = 1
            spectrum = _readout(self.base, self.guide, self.radius, weights, kernels) / self.area
            features.append([x[..., 0] for x in self.correction(spectrum, guide=True)])
        return [torch.stack(values, dim=-1) for values in zip(*features, strict=True)]


def _pair(ptycho, optics, region, factor, sampling, guide, scan_basis):
    with ptycho.select_region(region) as selected:
        baseline = selected.run(
            optics,
            method="phase",
            phase_damp=0.03,
            scan_basis=scan_basis,
            upsample=factor,
            compute_loss=False,
        )
        engine, settings, _ = direct_workflow._inputs(
            selected,
            baseline.aberrations,
            baseline.sample_tilt_mrad,
            baseline.depth_spread_nm,
            factor,
            baseline.reconstruction_options,
        )
        parity = (cp.asarray(engine.bf_inds_row) // 4 + cp.asarray(engine.bf_inds_col) // 4) % 2
        systems = [
            PreparedDirect.prepare(
                engine,
                settings,
                factor,
                complex_object=False,
                smoothness_inv_A=1.0,
                detector_weighting="uniform",
                detector_mask=parity == half,
            )
            for half in (0, 1)
        ]
        r0, r1, c0, c1 = region
        patch = guide[r0:r1, c0:c1]
        return [_Case(a, b, patch, sampling) for a, b in (systems, systems[::-1])]


def _fit_unet(training, validation, steps, trace):
    network = ResidualUNet(8).to(training[0].base.device)
    nn.init.zeros_(network.last.weight)
    nn.init.zeros_(network.last.bias)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=1e-5)
    best, selected, saved = 0.0, 0, deepcopy(network.state_dict())
    checks = {0, 5, 10, 25, 50, 100, 200, 400, steps}
    began = time.perf_counter()
    for step in tqdm(range(steps + 1), desc="learned correction", leave=False):
        if step in checks or step % 100 == 0:
            with torch.no_grad():
                gain = float(
                    torch.stack([c.gain(c.unet_phase(network)) for c in validation]).mean()
                )
            if not math.isfinite(gain):
                raise ValueError(
                    "Nonfinite U-Net validation gain; check counts and optical calibration."
                )
            trace.append(dict(stage="unet", step=step, validation_gain=gain))
            if gain > best:
                best, selected, saved = gain, step, deepcopy(network.state_dict())
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        for case in training:
            # Same centered diffraction objective as the staged full-dose design.
            normalization = 2 * case.area**2 / (case.base.numel() ** 2 * case.scale)
            loss = -case.gain(case.unet_phase(network)) * case.power * normalization / case.sigma**2
            (loss / len(training)).backward()
        nn.utils.clip_grad_norm_(network.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    network.load_state_dict(saved)
    return network.requires_grad_(False).eval(), selected, time.perf_counter() - began


@torch.no_grad()
def _readout_terms(cases):
    hessian = torch.zeros((18, 18), dtype=torch.float64, device=cases[0].base.device)
    gradient = torch.zeros(18, dtype=torch.float64, device=cases[0].base.device)
    for case in cases:
        anchor_ft = torch.fft.fft2(case.anchor) / case.area
        for b, (h, rhs), feature in zip(case.blocks, case.targets, case.features, strict=True):
            h, rhs, f = h.cdouble(), rhs.cdouble(), feature.cdouble()
            base = anchor_ft[b.rows, b.cols].cdouble()
            residual = (h @ base[..., None])[..., 0] - rhs
            hessian += (f.conj().transpose(-2, -1) @ h @ f).real.sum(0) / case.power
            gradient += (f.conj().transpose(-2, -1) @ residual[..., None])[..., 0].real.sum(
                0
            ) / case.power
    return hessian.cpu().numpy(), gradient.cpu().numpy()


def _fit_readout(training, validation, kernels, trace):
    from scipy.optimize import minimize

    for case in [*training, *validation]:
        case.features = case.readout_features(kernels)
    h, b = _readout_terms(training)
    hv, bv = _readout_terms(validation)
    candidates = [(0.0, np.zeros(18), None)]
    for ridge in (0.0, 0.001, 0.01, 0.1, 1.0, 10.0):
        matrix = h + ridge * max(np.trace(h) / 18, 1e-18) * np.eye(18)
        scale = max(abs(matrix).max(), abs(b).max(), 1e-18)

        def objective(x):
            return (x @ matrix @ x + 2 * b @ x) / scale, 2 * (matrix @ x + b) / scale

        fit = minimize(
            objective,
            np.zeros(18),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-5.0, 5.0)] * 18,
            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 1000},
        )
        if not fit.success:
            raise ValueError(f"Annular readout fitting failed: {fit.message}")
        gain = -(fit.x @ hv @ fit.x + 2 * bv @ fit.x) / len(validation)
        trace.append(dict(stage="readout", step=ridge, validation_gain=float(gain)))
        candidates.append((gain, fit.x, ridge))
    gain, weights, ridge = max(candidates, key=lambda row: row[0])
    weights = torch.as_tensor(weights, device=training[0].base.device, dtype=torch.float32)
    for case in [*training, *validation]:
        case.anchor = case.anchor + case.image(
            [feature @ weights.cfloat() for feature in case.features]
        )
        del case.features
    return weights, ridge, gain


@torch.no_grad()
def _direction_cases(cases):
    for case in cases:
        anchor = torch.fft.fft2(case.anchor) / case.area
        context = _context(case.anchor, case.base)
        norm = (case.base.std() * math.sqrt(case.base.numel()) / case.area).clamp_min(1e-5)
        case.direction_parts = []
        for b, (h, rhs) in zip(case.blocks, case.targets, strict=True):
            a = anchor[b.rows, b.cols]
            args = _direction_input(
                a,
                case.base_ft[b.rows, b.cols],
                b.gram,
                b.rhs,
                b.prior,
                case.scale,
                case.qr[b.rows],
                case.qc[b.cols],
                case.radius[b.rows, b.cols].clamp_min(1e-5),
                [v[b.rows, b.cols] for v in context],
                norm,
            )
            gradient = (h @ a[..., None])[..., 0] - rhs
            case.direction_parts.append((*args, h, gradient))


def _direction_gain(case, head):
    change = torch.zeros((), device=case.base.device, dtype=torch.float64)
    for parts in case.direction_parts:
        for start in range(0, len(parts[0]), 2048):
            block = [x[start : start + 2048] for x in parts]
            d = head(*block[:6])
            h, g = block[6:]
            change += (2 * (d.conj() * g).real + (d.conj() * (h @ d[..., None])[..., 0]).real).sum(
                dtype=torch.float64
            )
    return -change / case.power


def _fit_head(training, validation, steps, seed, trace, label="correction direction"):
    device = training[0].base.device
    torch.manual_seed(seed)
    head = Direction().to(device)
    nn.init.zeros_(head.decode[-1].weight)
    nn.init.zeros_(head.decode[-1].bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.0001)
    best, selected, saved = 0.0, 0, deepcopy(head.state_dict())
    significant, last_change = 0.0, 0
    for step in tqdm(range(steps + 1), desc=label, leave=False):
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                gain = float(torch.stack([_direction_gain(c, head) for c in validation]).mean())
            if not math.isfinite(gain):
                raise ValueError("Nonfinite direction validation gain; check the fitted phase.")
            trace.append(dict(stage=f"direction-{seed}", step=step, validation_gain=gain))
            if gain > best + 1e-10:
                best, selected, saved = gain, step, deepcopy(head.state_dict())
            if gain > significant + 2e-8:
                significant, last_change = gain, step
            if step >= 600 and step - last_change >= 500:
                break
        if step == steps:
            break
        case = training[step % 2]
        optimizer.zero_grad(set_to_none=True)
        for parts in case.direction_parts:
            ids = torch.randint(len(parts[0]), (min(384, len(parts[0])),), device=device)
            block = [x[ids] for x in parts]
            d = head(*block[:6])
            h, g = block[6:]
            loss = (2 * (d.conj() * g).real + (d.conj() * (h @ d[..., None])[..., 0]).real).sum()
            (loss * len(parts[0]) / len(ids) * 1e5 / case.power).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    head.load_state_dict(saved)
    return head.requires_grad_(False).eval(), selected


def train(
    ptycho,
    optics,
    *,
    steps,
    direction_steps,
    patch_size,
    seed,
    upsample,
    annulus_mrad,
    scan_basis,
):
    from quantem.gpu.direct_ptycho.neural_phase import NeuralPhaseModel

    if ptycho.backend != "cuda" or ptycho._input_dataset is None or ptycho.scan_region is not None:
        raise ValueError(
            "Load a full native counting acquisition on CUDA before fitting the neural model."
        )
    for name, value in (
        ("steps", steps),
        ("direction_steps", direction_steps),
        ("patch_size", patch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    shape = (ptycho._input_dataset.shape[0], ptycho._input_dataset.shape[1])
    regions = _regions(shape, patch_size)
    factor = ptycho._resolve_upsample(upsample)
    sampling = ptycho._output_sampling(factor)
    pixel_margin = max(np.broadcast_to(ptycho.detector_sampling_mrad, (2,)))
    # The annulus starts clear of the bright-field disk by two detector pixels, and grows
    # with the pixel size when one pixel is a large fraction of the convergence angle.
    inner = max(1.05 * ptycho.semiangle_mrad, ptycho.semiangle_mrad + 2 * pixel_margin)
    outer = max(1.5 * ptycho.semiangle_mrad, inner + 2 * pixel_margin)
    limits = annular_limits(annulus_mrad or (inner, outer))
    if limits[0] < ptycho.semiangle_mrad + pixel_margin:
        raise ValueError(
            "Choose an annulus outside the bright-field disk plus one detector pixel for detector-held-out training."
        )
    device = torch.device("cuda", ptycho._device)
    with cp.cuda.Device(device.index), torch.random.fork_rng(devices=[device.index]):
        began = time.perf_counter()
        torch.manual_seed(seed)
        guide = torch.from_dlpack(annular_counts(ptycho, ptycho._input_dataset, limits)).float()
        training, validation = [
            _pair(ptycho, optics, region, factor, sampling, guide, scan_basis) for region in regions
        ]
        trace = []
        network, unet_step, unet_s = _fit_unet(training, validation, steps, trace)
        with torch.no_grad():
            for case in [*training, *validation]:
                case.anchor = case.unet_phase(network)
        kernels = readout_kernels().to(device)
        weights, ridge, _ = _fit_readout(training, validation, kernels, trace)
        _direction_cases([*training, *validation])
        heads, head_steps = [], []
        for index, head_seed in enumerate((seed + 13, seed + 37), start=1):
            head, selected = _fit_head(training, validation, direction_steps, head_seed, trace,
                                       label=f"correction direction {index} of 2")
            heads.append(head)
            head_steps.append(selected)
        torch.cuda.synchronize(device)
        record = dict(
            recipe="staged",
            history=trace,
            selected_step=unet_step,
            selected_direction_steps=head_steps,
            selected_readout_ridge=ridge,
            train_region=regions[0],
            validation_region=regions[1],
            seed=seed,
            max_steps=steps,
            max_direction_steps=direction_steps,
            training_s=time.perf_counter() - began,
            unet_s=unet_s,
            electron_fraction=1.0,
            detector_holdout="complementary 4x4 bright-field blocks",
            conditioning="Fixed full-data optics; measured disjoint annulus shared between detector subsets.",
            full_data_note="Inference noise channel 1/sqrt(2) extrapolates from detector-half training at 1. Validate the full-dose output independently.",
        )
        return NeuralPhaseModel(
            dict(
                schema=NeuralPhaseModel._SCHEMA,
                width=8,
                unet={k: v.cpu() for k, v in network.state_dict().items()},
                readout=weights.cpu(),
                directions=[{k: v.cpu() for k, v in h.state_dict().items()} for h in heads],
                voltage_kV=ptycho.voltage_kV,
                semiangle_mrad=ptycho.semiangle_mrad,
                object_sampling_A=sampling,
                upsample=factor,
                annulus_mrad=limits,
                training_record=record,
            )
        )


@torch.inference_mode()
def held_scores(model, ptycho, optics, region, scan_basis):
    """Score a disjoint spatial region with fresh detector-block predictions."""
    from quantem.gpu.direct_ptycho._phase_training import _evaluation_region, _PhaseTable
    from quantem.gpu.direct_ptycho._phase_refinement import refine

    model._check(ptycho, model._factor)
    if region is None:
        if not {"train_region", "validation_region"} <= model.training_record.keys():
            raise ValueError(
                "Specify an unused scan_region; this model has no recorded fitting regions."
            )
        region = _evaluation_region(ptycho.scan_shape, model.training_record)
    # A requested test region must not overlap data used to fit or select weights.
    for key in ("train_region", "validation_region"):
        used = model.training_record.get(key)
        if used is not None and not (
            region[1] <= used[0]
            or region[0] >= used[1]
            or region[3] <= used[2]
            or region[2] >= used[3]
        ):
            raise ValueError(
                "Choose a test scan_region disjoint from the training and selection patches."
            )
    device = torch.device("cuda", ptycho._device)
    with cp.cuda.Device(device.index):
        guide = torch.from_dlpack(
            annular_counts(ptycho, ptycho._input_dataset, model.annulus_mrad)
        ).float()
        cases = _pair(ptycho, optics, region, model._factor, model._sampling, guide, scan_basis)
        net = model._on_device(device)
        scores = {"phase": [], "neural_phase": []}
        for case in cases:
            neural = refine(
                case.source,
                case.base,
                case.guide,
                model._sampling,
                net["unet"],
                net["heads"],
                net.readout,
                net.kernels,
                noise_level=1.0,
            )
            direct_error = case.target.phase_error(cp.from_dlpack(case.base))
            scores["phase"].append(direct_error)
            scores["neural_phase"].append(direct_error - float(case.gain(neural)))
        table = _PhaseTable.from_dict(
            scores, orient="index", columns=["A predicts B", "B predicts A"]
        )
        table["Mean held error"] = table.mean(axis=1)
        table.index.name = "Reconstruction"
        table.attrs.update(
            scan_region=region,
            partition="detector blocks",
            phase_damp=0.03,
            conditioning="Fixed full-data optical fit and shared disjoint annular guide.",
        )
        return table
