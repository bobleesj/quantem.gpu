"""One notebook phase view, updated after initialization and final refinement."""


class FitDisplay:
    """Render only in a live notebook; command-line fitting stays display-free."""

    def __init__(self, ssb, *, enabled: bool):
        self.ssb = ssb
        self.handle = None
        self.display = None
        if not enabled:
            return
        try:
            from IPython import get_ipython
            from IPython.display import display
        except ImportError:
            return
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            self.display = display

    def update(self, fit, *, final: bool) -> None:
        """Show Auto phase without fitting again or replacing the returned native fit."""
        if self.display is None:
            return
        try:
            image = self.ssb.reconstruct(
                fit, upsampling_factor="auto", compute_loss=False, verbose=False,
            )
            figure = image.show(show_parameters=False)
            figure.suptitle("Final fifth-order fit" if final else "Initial fit · refining through fifth order…")
            if self.handle is None:
                self.handle = self.display(figure, display_id=True)
            else:
                self.handle.update(figure)
        finally:
            # Rendering must not replace the native fit, loss or calibration.
            self.ssb._accept_result(fit)
