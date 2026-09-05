"""MPS/Metal SSB FFT configuration for 1024x1024 scans."""

from .common import MPSFFTConfig

CONFIG = MPSFFTConfig(
    size=1024,
    digit_reverse_define=(
        "#define BITREV4_10(x) ((((x) & 0x003u) << 8) | (((x) & 0x00Cu) << 4) | "
        "(((x) & 0x030u)) | (((x) & 0x0C0u) >> 4) | (((x) & 0x300u) >> 8))\n"
        "#define DIGITREVN(x) BITREV4_10(x)"
    ),
    digit_reverse_undef="#undef BITREV4_10\n#undef DIGITREVN",
    radix4_max=1024,
    has_final_radix2=False,
)

__all__ = ["CONFIG"]
