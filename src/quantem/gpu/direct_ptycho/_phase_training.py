"""Fit a portable phase model to disjoint electron counts from one native acquisition."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch import nn

from quantem.gpu import detector
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.direct_ptycho._annular_detector import annular_limits, annular_mask, measured_validity
from quantem.gpu.direct_ptycho._phase_networks import Direction, ResidualUNet
from quantem.gpu.direct_ptycho._phase_refinement import readout_kernels, refine
from quantem.gpu.direct_ptycho.neural_phase import NeuralPhaseModel


class _PhaseTable(pd.DataFrame):
    """Preserve numeric scores for analysis and show small differences in notebooks."""

    @property
    def _constructor(self):
        return _PhaseTable

    def _repr_html_(self) -> str:
        return self.to_html(float_format="{:.8f}".format)


def _evaluation_region(shape, record):
    """Find the first native 128-pixel patch clear of both fitting regions."""
    size = 128
    used = [record[key] for key in ("train_region", "validation_region") if key in record]
    for row in sorted({0, *(r[1] for r in used)}):
        for col in sorted({0, *(r[3] for r in used)}):
            end_row, end_col = row + size, col + size
            if end_row <= shape[0] and end_col <= shape[1] and all(
                end_row <= r0 or row >= r1 or end_col <= c0 or col >= c1
                for r0, r1, c0, c1 in used
            ):
                return row, end_row, col, end_col
    raise ValueError("No unused 128 × 128 patch fits. Score a separate acquisition and supply scan_region.")


def _regions(shape, size):
    rows, cols = shape
    if size not in (128, 256, 512, 1024):
        raise ValueError(
            "Use a supported native CUDA scan patch (128, 256, 512 or 1024) to avoid FFT padding."
        )
    if rows >= size and cols >= 2 * size:
        row = (rows - size) // 2
        left, right = (cols // 2 - size) // 2, cols // 2 + (cols - cols // 2 - size) // 2
        return [(row, row + size, c, c + size) for c in (left, right)]
    if cols >= size and rows >= 2 * size:
        col = (cols - size) // 2
        top, bottom = (rows // 2 - size) // 2, rows // 2 + (rows - rows // 2 - size) // 2
        return [(r, r + size, col, col + size) for r in (top, bottom)]
    if rows >= size and cols >= size:
        # One patch fits: train and hold out complementary electron halves of the
        # same detector patch. Weaker than two spatial regions, and recorded.
        row, col = (rows - size) // 2, (cols - size) // 2
        region = (row, row + size, col, col + size)
        return [region, region]
    if rows >= 64 and cols >= 64:
        # Smaller scan: the whole field is the single patch; the backend pads it
        # to the next FFT size, as it does for the phase solve.
        region = (0, rows, 0, cols)
        return [region, region]
    raise ValueError(
        f"A {size}-pixel training patch needs at least {size} rows and {size} columns; "
        f"two spatial regions need {size} rows and {2 * size} columns. Got {shape}."
    )


@dataclass
class _Half:
    equations: object
    phase: torch.Tensor
    guide: torch.Tensor
    complex_equations: object | None = None
    complex_phase: torch.Tensor | None = None
    log_amplitude: torch.Tensor | None = None


def _count_pair(ptycho, fit, region, factor, valid, annulus, generator, *, include_complex=False):
    scan = ptycho._input_dataset
    counts = scan.read(scan_region=region, output="torch")
    if counts.dtype not in (torch.uint8, torch.uint16, torch.uint32):
        raise ValueError(
            "Electron thinning needs native unsigned electron counts; load the counting acquisition losslessly."
        )
    if counts.dtype == torch.uint32 and int(cp.from_dlpack(counts).max()) > 2**24:
        raise ValueError("Count values exceed exact float32 integer precision for electron splitting.")
    raw = counts.float()
    if valid is not None:
        raw *= torch.as_tensor(valid, device=raw.device)
    first = torch.binomial(raw, raw.new_tensor(0.5).expand_as(raw), generator=generator)
    second = raw - first
    del raw, counts
    options = dict(
        backend="cuda",
        voltage_kV=ptycho.voltage_kV,
        semiangle_mrad=ptycho.semiangle_mrad,
        scan_sampling_A=ptycho.scan_sampling_A,
        detector_sampling_mrad=ptycho.detector_sampling_mrad,
        bright_field_center_px=ptycho.bright_field_center_px,
        bright_field_radius_px=ptycho.bright_field_radius_px,
        scan_detector_rotation_deg=fit.scan_detector_rotation_deg,
        scan_detector_rotation_flipped=fit.scan_detector_rotation_flipped,
    )
    pair = []
    for half in (first, second):
        with type(ptycho)(cp.from_dlpack(half), **options) as session:
            result = session.run(
                fit, method="phase", upsample=factor, compute_loss=False, verbose=False
            )
            equations = session._prepared_direct[1]["phase"]
            phase = torch.from_dlpack(result.phase).detach()
            item = _Half(equations, phase, half[:, :, annulus].sum(-1))
            if include_complex:
                result = session.run(
                    fit, method="complex", upsample=factor, compute_loss=False, verbose=False
                )
                item.complex_equations = session._prepared_direct[1]["complex"]
                item.complex_phase = torch.from_dlpack(result.phase)
                item.log_amplitude = torch.from_dlpack(result.amplitude).log()
        pair.append(item)
    return pair


def _gain(estimate, base, target):
    """Normalized error reduction against the other electron half; DC is unobserved."""
    spectrum = torch.fft.fft2(estimate) / target.factor**2
    base_spectrum = torch.fft.fft2(base) / target.factor**2
    loss = estimate.new_zeros((), dtype=torch.float64)
    for group in target.groups:
        rows, cols, gram, rhs = [
            torch.from_dlpack(x) for x in (group.rows, group.cols, group.gram, group.rhs)
        ]
        x = base_spectrum[rows, cols].to(gram.dtype)
        delta = spectrum[rows, cols].to(gram.dtype) - x
        gradient = (gram @ x[..., None])[..., 0] - rhs
        loss += (
            2 * (delta.conj() * gradient).real + (delta.conj() * (gram @ delta[..., None])[..., 0]).real
        ).sum()
    power = torch.from_dlpack(target.signal_power)
    if not bool(power > 0):
        raise ValueError(
            "Held counts carry no non-DC signal; use a different region or a larger counting scan."
        )
    return -loss / power


def _detector_mask(ptycho, limits, device):
    det = detector.prepare(ptycho._input_dataset)
    try:
        valid = measured_validity(ptycho._input_dataset, det.detector_validity)
    finally:
        det.close()
    mask = annular_mask(ptycho, ptycho._input_dataset.shape[-2:], limits, valid)
    return valid, torch.as_tensor(mask, device=device)


def train(ptycho, fit, checkpoint, steps, patch_size, seed, upsample, annulus_mrad):
    scan = ptycho._input_dataset
    if ptycho.backend != "cuda" or scan is None or ptycho.scan_region is not None:
        raise ValueError("Open a full native counting acquisition as a CUDA dataset before fitting.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int):
        raise ValueError("patch_size must be an integer number of scan pixels.")
    regions = _regions(scan.shape[:2], patch_size)
    factor = ptycho._resolve_upsample(upsample)
    sampling = ptycho._output_sampling(factor)
    limits = (
        (1.05 * ptycho.semiangle_mrad, 1.5 * ptycho.semiangle_mrad)
        if annulus_mrad is None else annular_limits(annulus_mrad)
    )
    device = torch.device("cuda", ptycho._device)
    with cp.cuda.Device(device.index), torch.random.fork_rng(devices=[device.index]):
        torch.default_generator.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        valid, annulus = _detector_mask(ptycho, limits, device)
        training, validation = [
            _count_pair(ptycho, fit, r, factor, valid, annulus, generator) for r in regions
        ]
        unet = ResidualUNet(8).to(device)
        head = Direction().to(device)
        nn.init.zeros_(unet.last.weight)
        nn.init.zeros_(unet.last.bias)
        nn.init.zeros_(head.decode[-1].weight)
        nn.init.zeros_(head.decode[-1].bias)
        readout = nn.Parameter(torch.zeros(18, device=device))
        kernels = readout_kernels().to(device)
        parameters = [*unet.parameters(), *head.parameters(), readout]
        optimizer = torch.optim.Adam(parameters, lr=0.001)

        def evaluate(pair, side):
            source = pair[side]
            prediction = refine(
                source.equations, source.phase, source.guide, sampling, unet, [head], readout, kernels
            )
            return _gain(prediction, source.phase, pair[1 - side].equations)

        best_gain = 0.0
        selected = 0
        best = (deepcopy(unet.state_dict()), deepcopy(head.state_dict()), readout.detach().clone())
        trace = [dict(step=0, validation_gain=0.0)]
        for step in tqdm(range(1, steps + 1), desc="correction", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss = -evaluate(training, step % 2) * 1e5
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0 or step == steps:
                with torch.no_grad():
                    gain = float((evaluate(validation, 0) + evaluate(validation, 1)) / 2)
                if not np.isfinite(gain):
                    raise ValueError(
                        "Nonfinite held error during training; check calibration and electron counts."
                    )
                trace.append(dict(step=step, validation_gain=gain))
                if gain > best_gain:
                    selected, best_gain = step, gain
                    best = (
                        deepcopy(unet.state_dict()),
                        deepcopy(head.state_dict()),
                        readout.detach().clone(),
                    )
        training_record = dict(
            history=trace,
            selected_step=selected,
            seed=seed,
            train_region=regions[0],
            validation_region=regions[1],
            electron_fraction=0.5,
            max_steps=steps,
        )
        model = NeuralPhaseModel(dict(
            schema=NeuralPhaseModel._SCHEMA, width=8,
            unet={k: v.cpu() for k, v in best[0].items()}, readout=best[2].cpu(),
            directions=[{k: v.cpu() for k, v in best[1].items()}],
            voltage_kV=ptycho.voltage_kV, semiangle_mrad=ptycho.semiangle_mrad,
            annulus_mrad=limits, object_sampling_A=sampling, upsample=factor,
            training_record=training_record,
        ))
        if checkpoint is not None:
            model.save(checkpoint)
        return model


def _joint_error(phase, log_amplitude, target):
    phase_ft = torch.fft.fft2(phase) / target.factor**2
    amplitude_ft = torch.fft.fft2(log_amplitude) / target.factor**2
    power = torch.from_dlpack(target.signal_power)
    if not bool(power > 0):
        raise ValueError("No non-DC signal in this held patch; choose a region with measured contrast.")
    loss = power.clone()
    for group in target.groups:
        rows, cols, gram, rhs = [
            torch.from_dlpack(x) for x in (group.rows, group.cols, group.gram, group.rhs)
        ]
        value = torch.cat((amplitude_ft[rows, cols], phase_ft[rows, cols]), dim=-1).to(gram.dtype)
        loss += (
            (value.conj() * (gram @ value[..., None])[..., 0]).real - 2 * (value.conj() * rhs).real
        ).sum()
    return float(loss / power)


@torch.inference_mode()
def held_scores(model, ptycho, fit, region, seed):
    if ptycho.backend != "cuda" or ptycho._input_dataset is None or ptycho.scan_region is not None:
        raise ValueError("Use a full native counting acquisition on CUDA to evaluate electron halves.")
    model._check(ptycho, model._factor)
    rows, cols = ptycho._input_dataset.shape[:2]
    if region is None:
        if not {"train_region", "validation_region"} <= model.training_record.keys():
            raise ValueError("This checkpoint has no fitting regions. Specify scan_region on an independent field.")
        region = _evaluation_region((rows, cols), model.training_record)
    r0, r1, c0, c1 = region
    if not (
        0 <= r0 < r1 <= rows
        and 0 <= c0 < c1 <= cols
        and r1 - r0 == c1 - c0
        and r1 - r0 in (128, 256, 512, 1024)
    ):
        raise ValueError(f"Use a supported square scan_region within the measured field {rows} × {cols}.")
    device = torch.device("cuda", ptycho._device)
    with cp.cuda.Device(device.index):
        valid, annulus = _detector_mask(ptycho, model._annulus, device)
        generator = torch.Generator(device=device).manual_seed(seed)
        halves = _count_pair(
            ptycho, fit, region, model._factor, valid, annulus, generator, include_complex=True
        )
        net = model._on_device(device)
        scores = {"phase": [], "neural_phase": [], "complex_phase + amplitude": []}
        for source, target in (halves, halves[::-1]):
            neural = refine(
                source.equations,
                source.phase,
                source.guide,
                model._sampling,
                net["unet"],
                net["heads"],
                net.readout,
                net.kernels,
            )
            zeros = torch.zeros_like(neural)
            for name, phase, eta in (
                ("phase", source.phase, zeros),
                ("neural_phase", neural, zeros),
                ("complex_phase + amplitude", source.complex_phase, source.log_amplitude),
            ):
                scores[name].append(_joint_error(phase, eta, target.complex_equations))
    table = _PhaseTable.from_dict(scores, orient="index", columns=["A predicts B", "B predicts A"])
    table["Mean held error"] = table.mean(axis=1)
    table.index.name = "Reconstruction"
    table.attrs["scan_region"] = region
    return table


def plot_training(model, *, size):
    """Plot optimizer-selection histories with one panel per fitting stage."""
    import matplotlib.pyplot as plt
    from quantem.gpu.display._layout import panel_inches

    if not model.training_history:
        raise ValueError("This model has no recorded training history. Fit a model to record validation curves.")
    width = panel_inches(size)
    staged = model.training_record.get("recipe") == "staged"
    if staged:
        figure, grid = plt.subplots(2, 2, figsize=(2 * width, 1.45 * width), layout="constrained")
        axes = list(grid.flat)
        axes[3].set_visible(False)
    else:
        figure, axis = plt.subplots(figsize=(width, width * .7), layout="constrained")
        axes = [axis]
    for stage in dict.fromkeys(row.get("stage", "unet") for row in model.training_history):
        rows = [r for r in model.training_history if r.get("stage", "unet") == stage]
        index = 1 if stage == "readout" else 2 if stage.startswith("direction") else 0
        axis = axes[index]
        label = {"unet": "learned correction", "readout": "blend"}.get(stage, stage)
        selected = model.selected_step
        if stage == "readout":
            selected = model.training_record.get("selected_readout_ridge")
            axis.set_xscale("symlog", linthresh=0.001)
        elif stage.startswith("direction"):
            stages = list(dict.fromkeys(r["stage"] for r in model.training_history
                                        if r.get("stage", "").startswith("direction")))
            number = stages.index(stage)
            label = f"Head {number + 1}"
            selected = model.training_record["selected_direction_steps"][number]
        line, = axis.plot([r["step"] for r in rows], [r["validation_gain"] for r in rows], ".-", label=label)
        for row in rows:
            if row["step"] == selected:
                axis.plot(row["step"], row["validation_gain"], "*", color=line.get_color(),
                          markersize=12, label=f"Selected {selected:g}")
        axis.axhline(0, color=".7", lw=.7)
        axis.set(xlabel="Blending strength" if stage == "readout" else "Optimizer update",
                 ylabel="Match to measured data\n(higher is better)",
                 title="Learned correction vs direct phase" if index == 0 else "Added by the correction")
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("Training selection · higher is better · no correction stays a candidate")
    plt.close(figure)
    return figure
