"""Reuse Optuna 4.9 Gaussian terms without changing adaptive SSB trials.

The inverse CDF keeps Optuna's equations, iteration order and tolerance. Only
its scalar central log-CDF loop is compiled, without fast math. Extreme tails
use the original implementation to retain its floating-point behavior.
"""

# Adapted from Optuna's probability_distributions.py and _truncnorm.py.
# Copyright (c) 2018 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The truncated normal implementation includes code from SciPy:
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np
from numba import njit
from optuna.samplers._tpe import _truncnorm


@njit(cache=True)
def _central_log_cdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the original scalar math in float64 without Python dispatch."""
    result = np.empty_like(values)
    for index in range(values.size):
        value = values[index]
        argument = -value if value > 6.0 else value
        scaled = argument / 2**0.5
        if scaled < -1 / 2**0.5:
            probability = 0.5 * math.erfc(-scaled)
        elif scaled < 1 / 2**0.5:
            probability = 0.5 + 0.5 * math.erf(scaled)
        else:
            probability = 1 - 0.5 * math.erfc(scaled)
        # Extreme tails are filled by Optuna below. Do not take log(0).
        result[index] = (-probability if value > 6.0 else
                         math.log(probability) if value > -20.0 else np.nan)
    return result


def _inverse_log_cdf(values: np.ndarray) -> np.ndarray:
    """Keep Optuna's vector Newton iterations, including its stopping rule."""
    flipped = values > -1e-2
    target = values.copy()
    target[flipped] = np.log(-np.expm1(values[flipped]))
    result = np.empty_like(values)
    small = np.nonzero(target < -5)
    result[small] = -np.sqrt(-2 * (target[small] + _truncnorm._norm_pdf_logC))
    moderate = np.nonzero(target >= -5)
    result[moderate] = -_truncnorm._ndtri_exp_approx_C * np.log(np.expm1(-target[moderate]))
    for _ in range(100):
        log_cdf = _central_log_cdf(result)
        tail = result <= -20.0
        if np.any(tail):
            log_cdf[tail] = _truncnorm._log_ndtr(result[tail])
        log_pdf = -0.5 * result**2 - _truncnorm._norm_pdf_logC
        step = (log_cdf - target) * np.exp(log_cdf - log_pdf)
        result -= step
        if np.all(np.abs(step) < 1e-8 * np.abs(result)):
            break
    result[flipped] *= -1
    return result


class _CachedGaussianMixture:
    """Cache bounds and normalization for one unchanged Parzen density."""

    def __init__(self, original):
        self.weights = original.weights
        self.distributions = original.distributions
        self._mu = np.asarray([part.mu for part in self.distributions]).T
        self._sigma = np.asarray([part.sigma for part in self.distributions]).T
        self._lower = (np.asarray([part.low for part in self.distributions]) - self._mu) / self._sigma
        self._upper = (np.asarray([part.high for part in self.distributions]) - self._mu) / self._sigma
        self._log_mass = _truncnorm._log_gauss_mass(self._lower, self._upper)
        self._log_scale = np.log(self._sigma)
        self._log_weights = np.log(self.weights[None])
        self._log_lower = _truncnorm._log_ndtr(self._lower)
        self._log_negative_upper = _truncnorm._log_ndtr(-self._upper)

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        """Draw the same mixture choices and quantiles in the original order."""
        indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)
        lower = self._lower.T[:, indices]
        upper = self._upper.T[:, indices]
        quantiles = rng.uniform(low=0, high=1, size=lower.shape)
        mass = self._log_mass.T[:, indices]
        left = lower < 0
        right = ~left
        result = np.empty_like(quantiles)
        if np.any(left):
            result[left] = _inverse_log_cdf(np.logaddexp(
                self._log_lower.T[:, indices][left], np.log(quantiles[left]) + mass[left],
            ))
        if np.any(right):
            result[right] = -_inverse_log_cdf(np.logaddexp(
                self._log_negative_upper.T[:, indices][right], np.log1p(-quantiles[right]) + mass[right],
            ))
        result[quantiles == 0] = lower[quantiles == 0]
        result[quantiles == 1] = upper[quantiles == 1]
        result[lower == upper] = np.nan
        return (result * self._sigma.T[:, indices] + self._mu.T[:, indices]).T.copy()

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        """Score all candidates with unchanged float64 accumulation order."""
        standardized = (samples[:, None, :] - self._mu) / self._sigma
        density = _truncnorm._norm_logpdf(standardized) - self._log_mass - self._log_scale
        density = np.select(
            [self._lower == self._upper, (standardized < self._lower) | (standardized > self._upper)],
            [np.nan, -np.inf], default=density,
        )
        weighted = np.zeros((len(samples), len(self.weights)), dtype=np.float64)
        weighted += density.sum(axis=-1)
        weighted += self._log_weights
        maximum = weighted.max(axis=1)
        maximum[np.isneginf(maximum)] = 0
        with np.errstate(divide="ignore"):
            return np.log(np.exp(weighted - maximum[:, None]).sum(axis=1)) + maximum
