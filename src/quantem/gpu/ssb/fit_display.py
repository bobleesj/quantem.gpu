"""Notebook phase preview and an optional throttled convergence chart."""

import time


class FitDisplay:
    """Render only in a live notebook; command-line fitting stays display-free."""

    def __init__(self, ssb, *, enabled: bool, size: str | int = "medium", columns: int = 3,
                 show_convergence: bool = False,
                 scan_region: tuple[int, int, int, int] | None = None):
        from quantem.gpu.display._layout import panel_inches, resolve_columns

        self.width = panel_inches(size)
        self.size = size
        self.columns = resolve_columns(3, columns=columns)
        self.enabled = enabled
        self.show_convergence = show_convergence
        self.ssb = ssb
        self.scan_region = scan_region
        self.handle = None
        self.convergence_handle = None
        self.display = None
        self.figure = None
        self.last_update = 0.0
        self.display_s = 0.0
        self.scores = []
        self.evaluations = []
        if not (enabled or show_convergence):
            return
        try:
            from IPython import get_ipython
            from IPython.display import display
        except ImportError:
            return
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            self.display = display

    def progress(self, evaluation: int, agreement: float | None = None, *,
                 stage: str = "Preparing initial fit…", final: bool = False) -> None:
        """Reuse optimizer scalars, refreshing at most twice per second."""
        if self.display is None or not self.show_convergence:
            return
        if agreement is not None:
            self.evaluations.append(evaluation)
            self.scores.append(float(agreement))
        if not final and time.perf_counter() - self.last_update < 0.5:
            return
        started = time.perf_counter()
        import matplotlib.pyplot as plt

        if self.figure is None:
            self.figure, self.axis = plt.subplots(figsize=(self.width, self.width * 2 / 3))
            self.line, = self.axis.plot([], [], color="#27658b")
            self.axis.set_xlabel("Fifth-order objective evaluations")
            self.axis.set_ylabel("Best normalized model agreement ↑")
            self.axis.grid(alpha=0.2)
        self.line.set_data(self.evaluations, self.scores)
        self.axis.relim()
        self.axis.autoscale_view()
        self.axis.set_title(stage)
        self.figure.tight_layout()
        if self.convergence_handle is None:
            self.convergence_handle = self.display(self.figure, display_id=True)
        else:
            self.convergence_handle.update(self.figure)
        # Prevent inline Matplotlib from emitting an extra figure at cell end.
        plt.close(self.figure)
        self.last_update = time.perf_counter()
        self.display_s += self.last_update - started

    def update(self, fit, *, final: bool) -> None:
        """Compare final direct methods without refitting or replacing the native fit."""
        if self.display is None or not self.enabled:
            return
        try:
            image = self.ssb.run(
                fit, upsampling_factor="auto", compute_loss=False, verbose=False,
            )
            comparisons = ()
            if final and self.ssb.backend == "cuda" and self.ssb._streamed is None:
                # Prepare only after fitting. Finish with phase so subsequent
                # regularization exploration can reuse its prepared equations.
                complex_image = self.ssb.run(
                    fit, method="complex", upsampling_factor="auto", compute_loss=False,
                )
                phase_image = self.ssb.run(
                    fit, method="phase", upsampling_factor="auto", compute_loss=False,
                )
                comparisons = (phase_image, complex_image)
            if self.scan_region is not None:
                figure = _show_fit_region(
                    image, self.scan_region, self.ssb.input_scan_shape, width=self.width,
                    final=final, compare=comparisons, columns=self.columns,
                )
            elif comparisons:
                from .results import _phase_comparison

                figure = _phase_comparison(
                    image, *comparisons, axsize=(self.width, self.width),
                    cmap="gray", region=None, columns=self.columns,
                )
            else:
                figure = image.show(show_parameters=False, size=self.size)
                figure.suptitle("Final fifth-order fit" if final else "Initial fit · refining through fifth order…")
                # The image already has a multiline title. Reflow after adding
                # the fit heading so both titles have their own vertical space.
                figure.tight_layout(rect=(0, 0, 1, .9), pad=1.2)
            if self.handle is None:
                self.handle = self.display(figure, display_id=True)
            else:
                self.handle.update(figure)
        finally:
            # Rendering must not replace the native fit, loss or calibration.
            self.ssb._accept_result(fit)


def _show_fit_region(image, scan_region, scan_shape, *, width: float, final: bool,
                     compare=(), columns: int = 3):
    """Show full fields above matching fit-region zooms under one contrast scale."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from quantem.core.visualization import show_2d
    from quantem.gpu.display._layout import panel_figure, scalebar
    from quantem.gpu.ssb.results import _phase_title, host_array

    factor = image.upsampling_factor
    row_start, row_stop, col_start, col_stop = scan_region
    # Drop FFT padding from the overview: the box is in measured scan positions.
    results = (image, *compare)
    full = [host_array(result.phase)[:scan_shape[0] * factor, :scan_shape[1] * factor]
            for result in results]
    crops = [phase[row_start * factor:row_stop * factor, col_start * factor:col_stop * factor]
             for phase in full]
    sampling = image.object_sampling_A[1]
    full_labels = [_phase_title(result, area="Full field · fit region boxed") for result in results]
    crop_labels = [_phase_title(result, area="Fit-region zoom") for result in results]
    if compare:
        columns = min(columns, len(results))
        rows = 2 * ((len(results) + columns - 1) // columns)
        groups = []
        for start in range(0, len(results), columns):
            stop = start + columns
            groups.extend(((full[start:stop], full_labels[start:stop], True),
                           (crops[start:stop], crop_labels[start:stop], False)))
    elif columns == 1:
        rows, groups = 2, [(full, full_labels, True), (crops, crop_labels, False)]
    else:
        rows, columns = 1, 2
        groups = [(full + crops, full_labels + crop_labels, True)]
    title_lines = max(label.count("\n") + 1 for label in full_labels + crop_labels)
    figure, axes = panel_figure(rows, columns, width, width, title_lines=title_lines,
                                top_in=.5, bottom_in=1.0)
    figure.suptitle("Final regional aberration fit" if final else "Initial regional fit · refining through fifth order…",
                    y=1 - .1 / figure.get_figheight(), va="top")
    for row_axes, (frames, labels, overview) in zip(axes, groups, strict=True):
        show_2d(
            frames, title=labels, figax=(figure, row_axes[:len(frames)]), cmap="gray", cbar=True,
            norm={"interval_type": "manual", "vmin": image.phase_limits[0], "vmax": image.phase_limits[1]},
            scalebar=[scalebar(sampling, "Å", frame.shape[0], width) for frame in frames],
            title_fontsize=min(14, max(8, width * 2.2)), tight_layout=False,
        )
        for axis in row_axes[len(frames):]:
            axis.set_visible(False)
        if overview:
            for axis in row_axes[:min(len(results), len(frames))]:
                axis.add_patch(Rectangle(
                    (col_start * factor - .5, row_start * factor - .5),
                    (col_stop - col_start) * factor, (row_stop - row_start) * factor,
                    fill=False, edgecolor="#00D5FF", linewidth=2,
                ))
    figure.text(
        .5, .1 / figure.get_figheight(),
        f"Fit positions: rows {row_start}:{row_stop}, columns {col_start}:{col_stop}.\n"
        "Only boxed positions contribute to fitting; all previews apply those optics to the full acquisition.\n"
        "Shared phase contrast; this is not a full-field validation.",
        ha="center", va="bottom", fontsize=min(11, width * 2.2),
    )
    plt.close(figure)
    return figure
