"""Frozen networks that propose corrections to the current phase only."""

import torch
from torch import nn
from torch.nn import functional as F


class PhaseConv2d(nn.Conv2d):
    """Use full float32 convolution without changing process-wide TF32 settings."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._convolution(
            image,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            False,
            (0, 0),
            self.groups,
            False,
            False,
            True,
            False,
        )


class ResidualUNet(nn.Module):
    """Zero-mean phase residual from normalized phase and its noise level."""

    def __init__(self, width: int):
        super().__init__()

        def block(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                PhaseConv2d(inputs, outputs, 3, padding=1),
                nn.ReLU(),
                PhaseConv2d(outputs, outputs, 3, padding=1),
                nn.ReLU(),
            )

        sizes = [width * 2**index for index in range(4)]
        self.down = nn.ModuleList(block(a, b) for a, b in zip([2, *sizes[:-1]], sizes))
        self.reduce = nn.ModuleList(
            PhaseConv2d(b, a, 3, padding=1) for a, b in reversed(list(zip(sizes[:-1], sizes[1:])))
        )
        self.up = nn.ModuleList(block(2 * a, a) for a in reversed(sizes[:-1]))
        self.last = PhaseConv2d(width, 1, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        levels = []
        for index, block in enumerate(self.down):
            image = block(F.avg_pool2d(image, 2) if index else image)
            levels.append(image)
        for reduce, block, skip in zip(self.reduce, self.up, reversed(levels[:-1])):
            resized = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            image = block(torch.cat((reduce(resized), skip), dim=1))
        residual = self.last(image)
        return residual - residual.mean(dim=(-2, -1), keepdim=True)


class Direction(nn.Module):
    """Learn the complex direction of a bounded correction within one alias group."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(25, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.key = nn.Linear(24, 12)
        self.query = nn.Linear(24, 12)
        self.value = nn.Linear(24, 16)
        self.decode = nn.Sequential(nn.Linear(40, 24), nn.SiLU(), nn.Linear(24, 6))

    def forward(self, features, anchor, direct, response, coherence, radius):
        scale = (anchor.abs() + response.abs()).clamp_min(1e-12)
        cross = (response * anchor.conj()).real / scale.square().clamp_min(1e-12)
        hidden = self.encode(torch.cat((features, cross[..., None], (response.abs() / scale)[..., None]), -1))
        affinity = self.query(hidden) @ self.key(hidden).mT / 12**0.5
        attention = (affinity + coherence.clamp_min(1e-5).log()).softmax(dim=-1)
        weight = self.decode(torch.cat((hidden, attention @ self.value(hidden)), -1))
        innovation = response / scale * anchor.abs()
        difference = (direct - anchor) / (direct.abs() + anchor.abs()).clamp_min(1e-12) * anchor.abs()
        proposal = (
            anchor
            * (
                weight[..., 0]
                + 1j * (weight[..., 3] * features[..., 21] + weight[..., 4] * features[..., 22])
            )
            + weight[..., 1] * innovation
            + weight[..., 2] * difference
        )
        budget = 0.25 * anchor.abs()
        taper = 0.5 * (1 + torch.cos(torch.pi * ((radius - 1.0) / 0.5).clamp(0, 1)))
        return proposal * budget / (budget.square() + proposal.abs().square()).clamp_min(1e-25).sqrt() * taper
