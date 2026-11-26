import numpy as np
import pandas as pd
import pytest
from sklearn.kernel_ridge import KernelRidge

from cleanlab.regression.learn import CleanLearning


def _regression_data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(30, 3))
    targets = features @ np.array([0.8, -2, 1.1]) + rng.normal(scale=0.4, size=30)
    targets[::9] += 5
    return features, targets


@pytest.mark.parametrize("coarse_search_range", [[0.15], [0.05, 0.15, 0.25]])
@pytest.mark.parametrize("fine_search_size", [0, 1, 3])
def test_model_kwargs_used_throughout_outlier_fraction_search(
    coarse_search_range, fine_search_size
):
    features, targets = _regression_data()
    kwargs = {"sample_weight": 10.0}
    weighted = CleanLearning(model=KernelRidge(alpha=20), cv_n_folds=3, n_boot=0, seed=31)
    actual = weighted.find_label_issues(
        features,
        targets,
        model_kwargs=kwargs,
        uncertainty=1.0,
        coarse_search_range=coarse_search_range,
        fine_search_size=fine_search_size,
    )

    # A constant sample weight of ten is equivalent to dividing ridge alpha by ten.
    reference = CleanLearning(model=KernelRidge(alpha=2), cv_n_folds=3, n_boot=0, seed=31)
    expected = reference.find_label_issues(
        features,
        targets,
        uncertainty=1.0,
        coarse_search_range=coarse_search_range,
        fine_search_size=fine_search_size,
    )

    assert weighted.k == reference.k
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    assert kwargs == {"sample_weight": 10.0}


@pytest.mark.parametrize("include_aleatoric", [False, True])
def test_model_kwargs_used_during_uncertainty_estimation(include_aleatoric):
    features, targets = _regression_data()
    settings = dict(
        cv_n_folds=3, n_boot=2, include_aleatoric_uncertainty=include_aleatoric, seed=31
    )
    weighted = CleanLearning(model=KernelRidge(alpha=20), **settings)
    actual = weighted.find_label_issues(
        features, targets, model_kwargs={"sample_weight": 10.0}, coarse_search_range=[0.0]
    )

    reference = CleanLearning(model=KernelRidge(alpha=2), **settings)
    expected = reference.find_label_issues(features, targets, coarse_search_range=[0.0])

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["get_epistemic_uncertainty", "get_aleatoric_uncertainty"])
def test_public_uncertainty_methods_accept_model_kwargs(method):
    features, targets = _regression_data()
    weighted = CleanLearning(model=KernelRidge(alpha=20), cv_n_folds=3, n_boot=2, seed=31)
    actual = getattr(weighted, method)(features, targets, model_kwargs={"sample_weight": 10.0})

    reference = CleanLearning(model=KernelRidge(alpha=2), cv_n_folds=3, n_boot=2, seed=31)
    expected = getattr(reference, method)(features, targets)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class LegacyUncertaintyCleanLearning(CleanLearning):
    def get_epistemic_uncertainty(self, X, y, predictions=None):
        self.epistemic_called = True
        return super().get_epistemic_uncertainty(X, y, predictions)

    def get_aleatoric_uncertainty(self, X, residual):
        self.aleatoric_called = True
        return super().get_aleatoric_uncertainty(X, residual)


@pytest.mark.parametrize("fit_options", [{}, {"model_kwargs": None}, {"model_kwargs": {}}])
@pytest.mark.parametrize("include_aleatoric", [False, True])
def test_empty_model_kwargs_preserve_legacy_uncertainty_overrides(fit_options, include_aleatoric):
    features, targets = _regression_data()
    settings = dict(
        cv_n_folds=3, n_boot=2, include_aleatoric_uncertainty=include_aleatoric, seed=31
    )
    legacy = LegacyUncertaintyCleanLearning(model=KernelRidge(alpha=2), **settings)
    actual = legacy.find_label_issues(features, targets, coarse_search_range=[0.0], **fit_options)

    reference = CleanLearning(model=KernelRidge(alpha=2), **settings)
    expected = reference.find_label_issues(features, targets, coarse_search_range=[0.0])

    assert legacy.epistemic_called
    assert getattr(legacy, "aleatoric_called", False) == include_aleatoric
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
