"""Fit and reuse measurement-conditioned neural corrections for direct phase."""

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import torch

from quantem.gpu.direct_ptycho.persistence import _write_atomically
from quantem.gpu.direct_ptycho._phase_networks import Direction, ResidualUNet
from quantem.gpu.direct_ptycho._phase_refinement import readout_kernels, refine


if TYPE_CHECKING:
    from pandas import DataFrame

    from quantem.gpu.direct_ptycho.results import DirectPtychoResult
    from quantem.gpu.direct_ptycho.workflow import DirectPtycho


class NeuralPhaseModel:
    """Frozen U-Net, annular readout and bounded spectral correction for direct phase.

    A fitted model belongs to a calibrated acquisition regime. Fit it on one scan
    or load existing weights, then reuse those weights across frames. Each
    incoming scan supplies its own phase, annular image and diffraction equations.

    Examples
    --------
    >>> fit = ptycho.fit_aberrations(tilt=True)
    >>> model = NeuralPhaseModel.fit(ptycho, fit)
    >>> reconstruction = ptycho.reconstruct(fit, method="phase", refinement=model,
    ...                                    upsampling="auto")
    >>> neural_phase = reconstruction.neural_phase
    """

    _SCHEMA = "quantem.phase-refinement/1"

    def __init__(self, checkpoint: dict):
        if checkpoint.get("schema") != self._SCHEMA:
            raise ValueError(f"Expected a {self._SCHEMA} checkpoint; check its training/export version.")
        required = {
            "voltage_kV",
            "semiangle_mrad",
            "object_sampling_A",
            "upsampling",
            "annulus_mrad",
            "width",
            "readout",
            "directions",
            "unet",
        }
        if missing := required.difference(checkpoint):
            raise ValueError(f"Incomplete phase checkpoint; export missing fields {sorted(missing)}.")
        self._voltage = float(checkpoint["voltage_kV"])
        self._semiangle = float(checkpoint["semiangle_mrad"])
        self._sampling = tuple(checkpoint["object_sampling_A"])
        self._factor = checkpoint["upsampling"]
        self._annulus = tuple(checkpoint["annulus_mrad"])
        if (
            len(self._sampling) != 2
            or len(self._annulus) != 2
            or not np.isfinite([*self._sampling, *self._annulus, self._voltage, self._semiangle]).all()
            or min(*self._sampling, self._voltage, self._semiangle) <= 0
            or not 0 <= self._annulus[0] < self._annulus[1]
            or not isinstance(self._factor, int)
            or self._factor < 1
        ):
            raise ValueError(
                "Export positive physical sampling, voltage and convergence angle and a valid annulus."
            )
        readout = torch.as_tensor(checkpoint["readout"], dtype=torch.float32)
        if readout.shape != (18,) or not readout.isfinite().all():
            raise ValueError("Export all 18 finite trained annular and phase readout weights.")
        unet = ResidualUNet(checkpoint["width"])
        unet.load_state_dict(checkpoint["unet"])
        heads = torch.nn.ModuleList()
        for weights in checkpoint["directions"]:
            head = Direction()
            head.load_state_dict(weights)
            heads.append(head)
        if not heads:
            raise ValueError("Export at least one trained spectral direction network.")
        self._net = torch.nn.ModuleDict({"unet": unet, "heads": heads}).requires_grad_(False).eval()
        self._net.register_buffer("readout", readout)
        self._net.register_buffer("kernels", readout_kernels())
        if any(not p.isfinite().all() for p in self._net.parameters()):
            raise ValueError("The checkpoint contains nonfinite network weights; re-export the selected fit.")
        self.checkpoint_sha256: str | None = None
        self.training_record = checkpoint.get("training_record", {})
        self.training_history = tuple(self.training_record.get("history", ()))
        self.selected_step = self.training_record.get("selected_step")
        # A deterministic weights identity also covers models constructed in memory.
        digest = hashlib.sha256()
        for name, tensor in sorted(self._net.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self._weights_sha256 = digest.hexdigest()
        self._devices = {}
        self._lock = Lock()

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPhaseModel":
        """Read a versioned tensor-only checkpoint on the host before starting a series.

        Parameters
        ----------
        path : str or Path
            Checkpoint from training, with physical calibration and all three trained stages.

        Returns
        -------
        NeuralPhaseModel
            A read-only model. Weights are copied to the acquisition GPU on first use.
        """
        with Path(path).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            handle.seek(0)
            model = cls(torch.load(handle, map_location="cpu", weights_only=True))
        model.checkpoint_sha256 = digest
        return model

    @classmethod
    def export(
        cls,
        path: str | Path,
        *,
        unet: Mapping[str, torch.Tensor],
        readout: Sequence[float],
        directions: Sequence[Mapping[str, torch.Tensor]],
        voltage_kV: float,
        semiangle_mrad: float,
        annulus_mrad: tuple[float, float],
        object_sampling_A: tuple[float, float],
        upsampling: int,
        training_record: dict | None = None,
    ) -> None:
        """Package a fitted model for reuse after validation on held measurements.

        Parameters
        ----------
        path : str or Path
            Destination for the versioned weights and training calibration.
        unet : mapping
            State dictionary of the residual phase U-Net.
        readout : sequence of float
            Eighteen linear coefficients for the phase and annular feature bank.
        directions : sequence of mappings
            State dictionary for each selected direction network. Predictions are averaged.
        voltage_kV, semiangle_mrad : float
            Accelerating voltage and convergence semiangle of the fitted regime.
        annulus_mrad : pair of float
            Inner and outer collection angles used for the trained annular guide.
        object_sampling_A : pair of float
            Row and column phase pixel sizes at training.
        upsampling : int
            Phase-grid oversampling factor at training.
        training_record : dict, optional
            Selection history and training regions, stored with the selected weights.
        """
        if "down.0.0.weight" not in unet:
            raise ValueError("Supply the trained residual U-Net state dictionary.")
        checkpoint = dict(
            schema=cls._SCHEMA,
            width=unet["down.0.0.weight"].shape[0],
            unet={k: v.detach().cpu() for k, v in unet.items()},
            readout=torch.as_tensor(readout).float().cpu(),
            directions=[{k: v.detach().cpu() for k, v in head.items()} for head in directions],
            voltage_kV=float(voltage_kV),
            semiangle_mrad=float(semiangle_mrad),
            annulus_mrad=tuple(map(float, annulus_mrad)),
            object_sampling_A=tuple(map(float, object_sampling_A)),
            upsampling=upsampling,
        )
        if training_record is not None:
            checkpoint["training_record"] = training_record
        cls(checkpoint)
        _write_atomically(Path(path), "wb", lambda handle: torch.save(checkpoint, handle))

    def save(self, path: str | Path) -> None:
        """Optionally save the fitted model for later acquisitions.

        Fitting and reconstruction work entirely in memory. Saving is not
        required for the acquisition that trained the model.
        """
        self.export(
            path, unet=self._net["unet"].state_dict(), readout=self._net.readout,
            directions=[head.state_dict() for head in self._net["heads"]],
            voltage_kV=self._voltage, semiangle_mrad=self._semiangle,
            annulus_mrad=self._annulus, object_sampling_A=self._sampling,
            upsampling=self._factor, training_record=self.training_record,
        )

    @classmethod
    def fit(
        cls,
        ptycho: "DirectPtycho",
        optics: "DirectPtychoResult",
        *,
        checkpoint: str | Path | None = None,
        recipe: str = "staged",
        steps: int = 400,
        direction_steps: int = 2400,
        patch_size: int = 128,
        seed: int = 6611,
        upsampling: int | str = "auto",
        annulus_mrad: tuple[float, float] | None = None,
        scan_basis: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> "NeuralPhaseModel":
        """Fit a measured-data neural correction and return it in memory.

        Parameters
        ----------
        ptycho, optics
            Native CUDA counting acquisition and its calibrated optical fit.
        checkpoint : str or Path or None, optional
            Optional output file. No existing checkpoint is required. Omit it
            for an in-memory workflow; use ``model.save(path)`` later if useful.
        recipe : {"staged", "count_split"}, optional
            Default ``"staged"`` fits the residual U-Net, annular readout and
            two bounded direction heads in sequence. Training uses full-dose
            counts split into disjoint detector blocks, with a separate scan
            patch for selection. The earlier one-head, jointly trained
            electron-half recipe remains available as ``"count_split"``.
        steps : int, optional
            Maximum U-Net optimizer updates, default 400. Selection retains
            the best measured validation checkpoint, including zero correction.
        direction_steps : int, optional
            Maximum updates per direction head, default 2400. Staged training
            stops after five validation checks without a significant gain.
        patch_size : int, optional
            Training and selection patch side in native scan pixels, default
            128. Reconstruction still uses the full acquisition.
        seed : int, optional
            Reproducible initialization and sampled-frequency seed.
        upsampling : int or "auto", optional
            Output sampling shared by training and inference.
        annulus_mrad : pair of float or None, optional
            Measured guide angles. Default: 1.05–1.5 times the convergence
            semiangle. Staged fitting requires an annulus outside bright field.
        scan_basis : pair of pairs of float or None, optional
            Fixed affine scan calibration for the staged recipe.

        Returns
        -------
        NeuralPhaseModel
            Selected frozen model with stage histories and training timings.

        Notes
        -----
        No simulation or reference object is a training target. The selected
        architecture matches the staged full-dose design, but a new fit on
        different training/selection regions need not reproduce earlier weights
        or image quality. To retain the earlier training policy, use
        ``recipe="count_split", steps=60, seed=42``. Fixed full-data optics and correlated detector pixels
        make selection conditional. Evaluate on unused measurements separately.

        Examples
        --------
        >>> optics = ptycho.fit_aberrations(tilt=True)
        >>> model = NeuralPhaseModel.fit(ptycho, optics)
        >>> reconstruction = ptycho.reconstruct(optics, method="phase",
        ...                                    refinement=model, upsampling="auto")
        """
        if recipe == "staged":
            from quantem.gpu.direct_ptycho._phase_staged_training import train

            model = train(ptycho, optics, steps=steps, direction_steps=direction_steps,
                          patch_size=patch_size, seed=seed, upsampling=upsampling,
                          annulus_mrad=annulus_mrad, scan_basis=scan_basis)
        elif recipe == "count_split":
            if scan_basis is not None:
                raise ValueError("scan_basis requires recipe='staged'.")
            from quantem.gpu.direct_ptycho._phase_training import train

            model = train(ptycho, optics, None, steps, patch_size, seed, upsampling, annulus_mrad)
        else:
            raise ValueError("Choose recipe='staged' or recipe='count_split'.")
        if checkpoint is not None:
            model.save(checkpoint)
        return model

    def report(self) -> "DataFrame":
        """Return validation gains by training step and mark the selected checkpoint.

        Older checkpoints without a training history return an empty table.
        """
        from quantem.gpu.direct_ptycho._phase_training import _PhaseTable

        history = _PhaseTable(self.training_history)
        if "stage" in history:
            return history.rename(columns={"stage": "Stage", "step": "Step or ridge",
                                           "validation_gain": "Validation gain"}).set_index(["Stage", "Step or ridge"])
        history = history.reindex(columns=["step", "validation_gain"])
        history["Selected"] = history["step"] == self.selected_step
        return history.rename(columns={"step": "Step", "validation_gain": "Validation gain"}).set_index("Step")

    def show(self, *, size: str | int = "medium"):
        """Plot recorded stage validation gains without fitting again.

        U-Net gains are relative to direct phase. Readout and direction gains
        are incremental relative to their fixed input stage. Larger is better;
        these are selection scores, not independent final-test evidence.
        """
        from quantem.gpu.direct_ptycho._phase_training import plot_training

        return plot_training(self, size=size)

    def score(
        self,
        ptycho: "DirectPtycho",
        optics: "DirectPtychoResult",
        *,
        scan_region: tuple[int, int, int, int] | None = None,
        seed: int = 123,
        partition: str = "electron",
        scan_basis: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> "DataFrame":
        """Compare frozen reconstructions against held measured diffraction.

        Parameters
        ----------
        ptycho, optics
            CUDA session on a measured counting scan and its fixed calibrating fit.
        scan_region : tuple of int, optional
            Row start, row stop, column start, column stop. By default, select a
            128-pixel square outside the recorded training and validation patches.
            Supply an independent patch explicitly for older weights without region metadata.
        seed : int
            Seed for complementary binomial electron splits.

        partition : {"electron", "detector"}, optional
            Complementary electron counts (default) or full-dose detector
            blocks. Detector validation matches the staged training regime
            and compares direct and neural phase. Both condition on fixed optics.
        scan_basis : pair of pairs of float or None, optional
            Fixed scan geometry for detector-block evaluation.

        Returns
        -------
        pandas.DataFrame
            Normalized diffraction errors in both directions. Lower is better.
            Detector-block scoring uses the linear phase model. Electron scoring
            includes a complex-object control with phase and amplitude together.
            The evaluated coordinates are recorded in ``table.attrs["scan_region"]``.
        """
        if partition == "detector":
            from quantem.gpu.direct_ptycho._phase_staged_training import held_scores

            return held_scores(self, ptycho, optics, scan_region, scan_basis)
        if partition != "electron":
            raise ValueError("Choose partition='electron' or partition='detector'.")
        if scan_basis is not None:
            raise ValueError("Calibrated scan_basis scoring currently requires partition='detector'.")
        from quantem.gpu.direct_ptycho._phase_training import held_scores

        return held_scores(self, ptycho, optics, scan_region, seed)

    @property
    def input_phase_damping(self) -> float:
        """Phase damping used by this architecture before its learned correction."""
        return 0.03

    @property
    def annulus_mrad(self) -> tuple[float, float]:
        """Inner (inclusive) and outer (exclusive) collection angles of the trained guide."""
        return self._annulus

    def _check(self, session, factor: int) -> None:
        sampling = np.asarray(session.scan_sampling_A) / factor
        if session.backend != "cuda":
            raise ValueError("Frozen phase refinement currently requires a CUDA DirectPtycho session.")
        if (
            factor != self._factor
            or not np.allclose(sampling, self._sampling, rtol=1e-4)
            or not np.allclose(
                [session.voltage_kV, session.semiangle_mrad], [self._voltage, self._semiangle], rtol=1e-4
            )
        ):
            raise ValueError(
                f"Use weights trained for this sampling and optics. Checkpoint expects "
                f"{self._sampling} Å pixels at {self._factor}×, {self._voltage} kV and "
                f"{self._semiangle} mrad; got {tuple(sampling)} Å at {factor}×, "
                f"{session.voltage_kV} kV and {session.semiangle_mrad} mrad."
            )

    def _signature(self) -> dict:
        """Record the complete frozen model identity without private source paths."""
        return dict(
            architecture=self._SCHEMA, weights_sha256=self._weights_sha256,
            checkpoint_sha256=self.checkpoint_sha256, voltage_kV=self._voltage,
            semiangle_mrad=self._semiangle, object_sampling_A=self._sampling,
            upsampling=self._factor, annulus_mrad=self._annulus,
            input_phase_damping=self.input_phase_damping,
        )

    @staticmethod
    def _check_policy(policy) -> None:
        for name, expected in (("smoothness_inv_A", 1.0), ("detector_weighting", "uniform")):
            if policy.get(name) not in (None, expected):
                raise ValueError(f"The trained phase model requires {name}={expected!r}.")

    def _on_device(self, device):
        with self._lock:
            if device not in self._devices:
                self._devices[device] = deepcopy(self._net).to(device)
            return self._devices[device]

    @torch.inference_mode()
    def _apply(self, prepared, base, sampling, annular):
        """Apply frozen weights to explicit current-acquisition equations and images."""
        phase = torch.from_dlpack(base)
        if min(phase.shape) < 8:
            raise ValueError("The frozen U-Net needs at least eight output pixels on both axes.")
        model = self._on_device(phase.device)
        guide = torch.from_dlpack(annular).float()
        result = refine(
            prepared, phase, guide, sampling, model["unet"], model["heads"], model.readout, model.kernels
        )
        if not bool(result.isfinite().all()):
            raise ValueError("Frozen refinement produced nonfinite phase; check scan counts and calibration.")
        return result
