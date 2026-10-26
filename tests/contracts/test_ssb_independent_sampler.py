"""Fixed broad-search proposals must match stock Optuna's adaptive sequence."""

import numpy as np
import optuna
import pytest

from quantem.gpu.direct_ptycho._tpe_independent import _batch_quantiles, make_independent_sampler


@pytest.mark.parametrize("seed", [17, 42, 87])
def test_full_fifth_order_search_preserves_every_proposal(seed):
    if optuna.__version__ != "4.9.0":
        pytest.skip("Optimized sampler is qualified for Optuna 4.9.0")
    bounds = np.array([(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)] + [(-2, 2)] * 22)
    sequences = []
    for sampler in (optuna.samplers.TPESampler(seed=seed), make_independent_sampler(seed=seed, bounds=bounds)):
        study = optuna.create_study(direction="maximize", sampler=sampler)
        initial = np.zeros(28)
        initial[5] = 1
        study.enqueue_trial({f"p{i}": value for i, value in enumerate(initial)})
        proposals = []

        def objective(trial, proposals=proposals):
            point = np.array([trial.suggest_float(f"p{i}", low, high) for i, (low, high) in enumerate(bounds)])
            proposals.append(point)
            # Coupled nonlinear objective makes any proposal difference affect
            # later trial rankings, rather than replaying a fixed score list.
            return float(np.cos(point).sum() - 0.02 * np.dot(point, np.roll(point, 1)) - np.sum(point**2))

        study.optimize(objective, n_trials=400)
        sequences.append((np.array(proposals), np.array([trial.value for trial in study.trials])))
    for expected, actual in zip(*sequences):
        np.testing.assert_array_equal(actual, expected)


def test_quantiles_keep_each_parameter_convergence_and_tail_behavior():
    if optuna.__version__ != "4.9.0":
        pytest.skip("Optimized quantiles are qualified for Optuna 4.9.0")
    from optuna.samplers._tpe import _truncnorm

    rng = np.random.RandomState(4)
    quantiles = rng.uniform(size=(28, 24))
    lower = rng.uniform(-50, -0.01, size=quantiles.shape)
    upper = rng.uniform(0.01, 50, size=quantiles.shape)
    lower[0], upper[0] = -50, -25
    lower[1], upper[1] = 25, 50
    quantiles[2, :2] = [0, 1]
    expected = np.array([_truncnorm.ppf(q, a, b) for q, a, b in zip(quantiles, lower, upper)])
    np.testing.assert_array_equal(_batch_quantiles(quantiles, lower, upper), expected)


def test_other_optuna_version_keeps_stock_sampler(monkeypatch):
    monkeypatch.setattr(optuna, "__version__", "5.0.0")
    sampler = make_independent_sampler(seed=17, bounds=np.array([[-2, 2]]))
    assert type(sampler) is optuna.samplers.TPESampler
