"""MPS/Metal SSB FFT configuration for 128x128 scans."""

from .common import MPSFFTConfig

CONFIG = MPSFFTConfig(
    size=128,
    digit_reverse_define=(
        "#define BITREV4_6(x) ((((x) & 0x03u) << 4) | (((x) & 0x0Cu)) | "
        "(((x) & 0x30u) >> 4))\n"
        "#define DIGITREVN(x) ((((x) & 1u) << 6) | BITREV4_6((x) >> 1))"
    ),
    digit_reverse_undef="#undef BITREV4_6\n#undef DIGITREVN",
    radix4_max=64,
    has_final_radix2=True,
)

__all__ = ["CONFIG"]
