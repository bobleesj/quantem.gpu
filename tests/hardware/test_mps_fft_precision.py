"""Physical Metal Fourier parity, including complex gradient propagation."""

import pytest
import torch
from quantem.gpu.device.mps_fft import fft2

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Physical MPS required"
)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (6, 12),
        (16, 16),
        (48, 48),
        (96, 96),
        (128, 128),
        (192, 192),
        (192, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("norm", [None, "ortho", "forward"])
def test_forward_inverse_and_adjoint(shape, norm):
    """Use bounded mathematical inputs; no measured reconstruction runs on CPU."""
    generator = torch.Generator().manual_seed(21)
    values = torch.randn((2, *shape), dtype=torch.complex64, generator=generator)
    weights = torch.randn(values.shape, dtype=torch.complex64, generator=generator)
    for inverse in (False, True):
        reference = values.to(torch.complex128).requires_grad_(True)
        function = torch.fft.ifft2 if inverse else torch.fft.fft2
        expected = function(reference, norm=norm)
        (expected * weights.conj()).real.sum().backward()
        actual_input = values.to("mps").requires_grad_(True)
        actual = fft2(actual_input, inverse=inverse, norm=norm)
        (actual * weights.to("mps").conj()).real.sum().backward()
        assert actual.dtype == torch.complex64 and actual.device.type == "mps"
        # Final complex64 rounding only; preserve an absolute bound near zeros.
        for found, wanted in ((actual, expected), (actual_input.grad, reference.grad)):
            wanted = wanted.detach()
            torch.testing.assert_close(
                found.detach().cpu(),
                wanted.to(torch.complex64),
                rtol=1e-6,
                atol=1e-7 * wanted.abs().max().item(),
            )


def test_noncontiguous_conjugate_input_and_unsupported_precision():
    values = torch.randn((2, 48, 96), dtype=torch.complex64).to("mps")
    values = values.transpose(-1, -2).conj()
    expected = torch.fft.fft2(values.cpu().to(torch.complex128)).to(torch.complex64)
    torch.testing.assert_close(
        fft2(values).cpu(), expected, rtol=1e-6, atol=1e-7 * expected.abs().max().item()
    )
    with pytest.raises(TypeError, match="complex64 MPS"):
        fft2(values.real)
    with pytest.raises(ValueError, match="Unsupported"):
        fft2(values[:, :17])
