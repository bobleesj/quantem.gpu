"""Compose existing GPU widgets for a synchronized optical-phase explorer."""

import json
import math
from dataclasses import replace

from ipywidgets import HTML, VBox
from quantem.widget import Show2D, ShowPtycho

from quantem.gpu.direct_ptycho.probe import model_probe, probe_grid


class ProbeExplorer(VBox):
    """Keep the displayed probe synchronized with completed live reconstructions.

    The existing ShowPtycho owns controls and reconstruction. Show2D owns
    probe rendering. No optical fit, neural training or specimen model is
    changed by opening this view.
    """

    def __init__(self, session, **options):
        self.reconstruction = ShowPtycho(session, **options)
        self._reference = session._aberration_search
        if self._reference is None:
            self._reference = session.reconstruct(compute_loss=False)
        self._coefficients = None
        self.probe = None
        super().__init__()
        self.reconstruction.observe(self._update_probe, names="result_json")
        self._update_probe({"new": self.reconstruction.result_json})
        self.children = (
            self.reconstruction,
            HTML("<b>Model probe at the fitted reference plane</b><br>"
                 "Aberration controls change this phase plate. Damping changes only "
                 "the object reconstruction above. Phase uses a fixed −π to π scale; "
                 "pixels outside the illuminated aperture are shown as zero. "
                 "The aperture panel identifies that support."),
            self.probe,
        )

    def _update_probe(self, change):
        values = json.loads(change["new"] or "{}")
        controls = self.reconstruction
        coefficients = json.loads(controls.higher_order_aberrations_json or "{}")
        coefficients.update({key: values.get(key, self._reference.aberrations.get(key, 0.0))
                             for key in ("C10_nm", "C12_nm", "phi12_deg")})
        if coefficients == self._coefficients:
            return
        result = replace(self._reference, aberrations=coefficients)
        wave = model_probe(result, space="fourier")
        if result.backend == "cuda":
            import cupy as xp
        else:
            import mlx.core as xp
        amplitude = xp.abs(wave)
        illuminated = amplitude > xp.max(amplitude) * 1e-6
        phase = xp.where(illuminated, xp.angle(wave), 0)
        aperture = amplitude / xp.max(amplitude)
        sampling = probe_grid(result)[2]
        labels = ["Probe phase · radians", "Illuminated aperture · amplitude"]
        if self.probe is None:
            self.probe = Show2D(
                [phase, aperture], labels=labels, columns=2, size="medium",
                sampling=sampling, units="mrad", cmap=["twilight_shifted", "gray"],
                vmin=[-math.pi, 0], vmax=[math.pi, 1], auto_contrast=False,
                show_scale_bar=True, show_fft=False, verbose=False,
            )
        else:
            self.probe.pixel_size = sampling[1]
            self.probe.set_image([phase, aperture], labels=labels)
        self._coefficients = coefficients

    def close(self):
        """Detach the live observer and close both child viewers."""
        if self.comm is not None and hasattr(self, "reconstruction"):
            self.reconstruction.unobserve(self._update_probe, names="result_json")
            self.reconstruction.close()
        if self.probe is not None:
            self.probe.close()
        super().close()
