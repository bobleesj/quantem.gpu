"""MPS/Metal SSB FFT configuration for 256x256 scans."""

from .common import MPSFFTConfig

CONFIG = MPSFFTConfig(
    size=256,
    digit_reverse_define=(
        "#define BITREV4_8(x) ((((x) & 0x03u) << 6) | (((x) & 0x0Cu) << 2) | "
        "(((x) & 0x30u) >> 2) | (((x) & 0xC0u) >> 6))\n"
        "#define DIGITREVN(x) BITREV4_8(x)"
    ),
    digit_reverse_undef="#undef BITREV4_8\n#undef DIGITREVN",
    radix4_max=256,
    has_final_radix2=False,
)

__all__ = ["CONFIG"]
