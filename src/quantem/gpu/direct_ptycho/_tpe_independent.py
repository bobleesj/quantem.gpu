"""Batch independent Optuna proposals while preserving their numerical policy.

This internal sampler is limited to a fixed continuous space, sequential trials,
and Optuna 4.9. The scientific objective remains on its selected accelerator;
proposal generation and density ranking remain CPU float64 calculations.
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


import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState

if optuna.__version__ == "4.9.0":
    from optuna.samplers._tpe import _truncnorm
    from optuna.samplers._tpe.sampler import _split_trials

    from ._tpe_gaussian import _central_log_cdf


def _log_cdf(values):
    shape = values.shape
    flat = np.asarray(values, dtype=np.float64).ravel()
    result = _central_log_cdf(flat)
    tail = flat <= -20.0
    if np.any(tail):
        result[tail] = _truncnorm._log_ndtr(flat[tail])
    return result.reshape(shape)


def _inverse_cdf(values, groups):
    """Keep each parameter's original Newton stopping rule when batching."""
    flipped = values > -1e-2
    target = values.copy()
    target[flipped] = np.log(-np.expm1(values[flipped]))
    result = np.empty_like(values)
    small = np.nonzero(target < -5)
    result[small] = -np.sqrt(-2.0 * (target[small] + _truncnorm._norm_pdf_logC))
    moderate = np.nonzero(target >= -5)
    result[moderate] = -_truncnorm._ndtri_exp_approx_C * np.log(np.expm1(-target[moderate]))
    active = np.ones(values.size, dtype=bool)
    group_count = int(groups.max()) + 1
    for _ in range(100):
        current = result[active]
        log_cdf = _log_cdf(current)
        log_pdf = -0.5 * current**2 - _truncnorm._norm_pdf_logC
        step = (log_cdf - target[active]) * np.exp(log_cdf - log_pdf)
        current -= step
        result[active] = current
        converged = np.abs(step) < 1e-8 * np.abs(current)
        pending = np.zeros(group_count, dtype=bool)
        np.logical_or.at(pending, groups[active], ~converged)
        active &= pending[groups]
        if not np.any(active):
            break
    result[flipped] *= -1
    return result


def _batch_quantiles(quantiles, lower, upper):
    """Evaluate independent truncated-normal quantiles in original order."""
    groups = np.broadcast_to(np.arange(quantiles.shape[0])[:, None], quantiles.shape)
    left = lower < 0
    right = ~left
    mass = _truncnorm._log_gauss_mass(lower, upper)
    result = np.empty_like(quantiles)
    if np.any(left):
        values = _truncnorm._log_sum(_log_cdf(lower[left]), np.log(quantiles[left]) + mass[left])
        result[left] = _inverse_cdf(values, groups[left])
    if np.any(right):
        values = _truncnorm._log_sum(_log_cdf(-upper[right]), np.log1p(-quantiles[right]) + mass[right])
        result[right] = -_inverse_cdf(values, groups[right])
    result[quantiles == 0] = lower[quantiles == 0]
    result[quantiles == 1] = upper[quantiles == 1]
    result[lower == upper] = np.nan
    return result


class _IndependentSampler(optuna.samplers.TPESampler):
    """Reuse completed history and batch fixed-space inverse-CDF work.

    Candidate RNG draws, Gaussian models, CPU density ranking and acquisition
    argmax retain Optuna's order. Only independent inverse-CDF iterations are
    grouped, with a separate convergence mask for every parameter.
    """

    def __init__(self, *, seed, space):
        super().__init__(seed=seed)
        self._space = space
        self._study = None
        self._rows = {}
        self._columns = {}
        self._batch_key = None
        self._batch_values = {}

    def _get_internal_repr(self, trials, search_space):
        columns = self._columns.get(id(trials))
        if columns is not None and all(name in columns for name in search_space):
            return {name: columns[name] for name in search_space}
        return super()._get_internal_repr(trials, search_space)

    def _sample(self, study, trial, search_space):
        key = (study, trial.number)
        if self._batch_key == key:
            return {name: self._batch_values[name] for name in search_space}
        if len(search_space) != 1 or next(iter(search_space)) != next(iter(self._space)):
            return super()._sample(study, trial, search_space)
        trials = study._get_trials(deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED], use_cache=True)
        if study is not self._study:
            self._study, self._rows, self._columns = study, {}, {}
        names = tuple(self._space)
        for completed in trials:
            if completed.number in self._rows:
                continue
            if (completed.state != TrialState.COMPLETE or set(completed.params) != set(names)
                    or any(completed.distributions[name] != self._space[name] for name in names)):
                self._columns = {}
                return super()._sample(study, trial, search_space)
            self._rows[completed.number] = np.asarray([completed.params[name] for name in names], dtype=np.float64)
        below, above = _split_trials(study, trials, self._gamma(len(trials)), False)
        self._columns = {}
        for group in (below, above):
            matrix = np.asarray([self._rows[item.number] for item in group], dtype=np.float64).reshape(-1, len(names))
            self._columns[id(group)] = {name: matrix[:, i] for i, name in enumerate(names)}
        models, quantiles, means, scales, lows, highs = [], [], [], [], [], []
        for name, distribution in self._space.items():
            local = {name: distribution}
            low_model = self._build_parzen_estimator(study, local, below, handle_below=True)
            high_model = self._build_parzen_estimator(study, local, above, handle_below=False)
            mixture = low_model._mixture_distribution
            normal = mixture.distributions[0]
            indices = self._rng.rng.choice(len(mixture.weights), p=mixture.weights, size=self._n_ei_candidates)
            quantiles.append(self._rng.rng.uniform(low=0, high=1, size=(1, self._n_ei_candidates))[0])
            means.append(normal.mu[indices])
            scales.append(normal.sigma[indices])
            lows.append(distribution.low)
            highs.append(distribution.high)
            models.append((low_model, high_model))
        means, scales = np.asarray(means), np.asarray(scales)
        points = _batch_quantiles(np.asarray(quantiles), (np.asarray(lows)[:, None] - means) / scales,
                                 (np.asarray(highs)[:, None] - means) / scales) * scales + means
        self._batch_values = {}
        for i, ((name, distribution), (low_model, high_model)) in enumerate(zip(self._space.items(), models)):
            samples = {name: points[i]}
            scores = self._compute_acquisition_func(samples, low_model, high_model)
            selected = self._compare(samples, scores)[name]
            self._batch_values[name] = distribution.to_external_repr(selected)
        self._batch_key = key
        return {name: self._batch_values[name] for name in search_space}


def make_independent_sampler(*, seed: int, bounds: np.ndarray) -> optuna.samplers.TPESampler:
    """Build a fixed-space sampler, retaining stock Optuna on other versions.

    Parameters
    ----------
    seed : int
        Optuna seed, unchanged from the caller's search.
    bounds : numpy.ndarray
        Lower and upper bounds in parameter order, shape ``(n, 2)``. The caller
        must request ``p0`` through ``p<n-1>`` in order, in every trial, with one
        worker. This private factory is used by the fixed broad SSB search.
    """
    if optuna.__version__ != "4.9.0":
        return optuna.samplers.TPESampler(seed=seed)
    space = {f"p{i}": FloatDistribution(float(low), float(high)) for i, (low, high) in enumerate(bounds)}
    return _IndependentSampler(seed=seed, space=space)
