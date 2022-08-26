# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab import count
from cleanlab.outlier import OutOfDistribution


def make_data(
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[80, 40, 40],
    avg_trace=0.8,
    seed=1,  # set to None for non-reproducible randomness
):
    np.random.seed(seed=seed)

    m = len(means)  # number of classes
    n = sum(sizes)
    local_data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(m):
        local_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        test_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(local_data)
    true_labels_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    true_labels_test = np.hstack(test_labels)

    # Compute p(true_label=k)
    py = np.bincount(true_labels_train) / float(len(true_labels_train))

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=avg_trace * m,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Generate our noisy labels using the noise_matrix.
    s = generate_noisy_labels(true_labels_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))

    # Compute inverse noise matrix
    inv = count.compute_inv_noise_matrix(py, noise_matrix, ps=ps)

    # Estimate pred_probs
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        labels=s,
        cv_n_folds=3,
    )

    label_errors_mask = s != true_labels_train

    return {
        "X_train": X_train,
        "true_labels_train": true_labels_train,
        "X_test": X_test,
        "true_labels_test": true_labels_test,
        "labels": s,
        "label_errors_mask": label_errors_mask,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
        "inverse_noise_matrix": inv,
        "est_py": latent[0],
        "est_nm": latent[1],
        "est_inv": latent[2],
        "cj": latent[3],
        "pred_probs": latent[4],
        "m": m,
        "n": n,
    }


# Global to be used by all test methods. Only compute this once for speed.
data = make_data()


def test_wrong_info_assert_valid_inputs():
    features = data["X_train"]
    pred_probs = data["pred_probs"]

    OOD = OutOfDistribution()

    # TESTING: _assert_valid_inputs() asserts correct errors in fit
    try:
        OOD.fit()
    except Exception as e:
        assert "Not enough information to compute scores" in str(e)
        with pytest.raises(ValueError) as e:
            OOD.fit()
    try:
        OOD.fit(features=features, pred_probs=pred_probs)
    except Exception as e:
        assert "Cannot fit to object to both features and pred_probs" in str(e)
        with pytest.raises(ValueError) as e:
            OOD.fit(features=features, pred_probs=pred_probs)

    OOD = OutOfDistribution()

    features_flat = np.ravel(features)
    features_extra_dim = features[np.newaxis]
    try:
        OOD.fit(features=features_flat)
    except Exception as e:
        assert "array needs to be of shape (N, M)" in str(e)
        with pytest.raises(ValueError) as e:
            OOD.fit(features=features_flat)
    try:
        OOD.fit(features=features_extra_dim)
    except Exception as e:
        assert "array needs to be of shape (N, M)" in str(e)
        with pytest.raises(ValueError) as e:
            OOD.fit(features=features_extra_dim)

    # TODO: DO WE NEED TO TESTING: _assert_valid_inputs() asserts correct errors in score?


def test_wrong_info_fit_ood():
    features = data["X_train"]
    pred_probs = data["pred_probs"]
    labels = data["labels"]

    OOD = OutOfDistribution()

    ##### SHARED FIT wrong info
    # TESTING: wrong params
    with pytest.raises(ValueError) as e:  # testing wrong params
        OOD.fit(features=features, params={"strange_param": -1})
    with pytest.raises(ValueError) as e:  # testing wrong params
        OOD.fit(pred_probs=pred_probs, params={"strange_param": -1})

    # testing OOD param in fit for outlier detection
    mixed_params_dict = {"k": 10, "adjust_pred_probs": True}
    try:
        OOD.fit(features=features, params=mixed_params_dict)
    except Exception as e:
        assert "with features, passed in params dict can only contain" in str(e)
        with pytest.raises(ValueError) as e:
            OOD.fit(features=features, params=mixed_params_dict)

    # testing outlier param in fit for OOD detection
    try:
        OOD.fit(pred_probs=pred_probs, params=mixed_params_dict)
    except Exception as e:
        assert "with pred_probs, passed in params dict can only contain" in str(e)
        with pytest.raises(ValueError) as e:
            OOD.fit(features=features, params=mixed_params_dict)

    #### SCORE wrong info

    # TESTING: calling score before any fitting
    try:
        OOD.score(features=features)
    except Exception as e:
        assert "OOD Object needs to be fit on features first." in str(e)
        with pytest.raises(ValueError) as e:
            OOD.score(features=features)

    try:
        OOD.score(pred_probs=pred_probs)
    except Exception as e:
        assert (
            "OOD Object needs to be fit on pred_probs with param adjust_pred_probs=True first."
            in str(e)
        )
        with pytest.raises(ValueError) as e:
            OOD.score(pred_probs=pred_probs)

    # TESTING: calling scoring with opposite fitting
    OOD_outlier = OutOfDistribution()
    OOD_outlier.fit(features=features)
    try:
        OOD_outlier.score(pred_probs=pred_probs)
    except Exception as e:
        assert (
            "OOD Object needs to be fit on pred_probs with param adjust_pred_probs=True first."
            in str(e)
        )
        with pytest.raises(ValueError) as e:
            OOD_outlier.score(pred_probs=pred_probs)

    OOD_ood = OutOfDistribution()
    OOD_ood.fit(pred_probs=pred_probs, labels=labels)
    try:
        OOD_ood.score(features=features)
    except Exception as e:
        assert "OOD Object needs to be fit on features first." in str(e)
        with pytest.raises(ValueError) as e:
            OOD_ood.score(features=features)

    # # TODO FIX THIS! get_ood_thresholds returns 1-2 variables this does not work
    # # TESTING: calling ood after fitting but without adjust_pred_probs
    # OOD_ood = OutOfDistribution()\
    # OOD_ood.fit(pred_probs=pred_probs, params={"adjust_pred_probs": False})
    # try:
    #     OOD_ood.score(features=features)
    # except Exception as e:
    #     assert "OOD Object needs to be fit on pred_probs with param adjust_pred_probs=True first." in str(e)
    #     with pytest.raises(ValueError) as e:
    #         OOD_ood.score(features=features)


def test_params_logic():
    features = data["X_train"]
    pred_probs = data["pred_probs"]

    OOD = OutOfDistribution()

    # TESTING: params dict is a copy
    params_dict = {"k": 10, "t": 5}
    OOD.fit(features=features, params=params_dict)
    ood_params = OOD.params
    params_dict = params_dict.update({"k": 20})
    assert ood_params == OOD.params

    # test calling functions with different params performs differently


def test_class_public_func():
    features = data["X_test"]
    pred_probs = data["pred_probs"]
    labels = data["true_labels_test"]

    #### TESTING FIT:
    OOD_ood = OutOfDistribution()
    OOD_ood.fit(pred_probs=pred_probs, labels=labels)

    OOD_outlier = OutOfDistribution()
    OOD_outlier.fit(features=features)

    assert OOD_ood.confident_thresholds is not None and OOD_ood.knn is None
    assert OOD_outlier.knn is not None and OOD_outlier.confident_thresholds is None
    assert OOD_ood.params is not None and OOD_outlier.params is not None

    #### TESTING SCORE
    ood_score = OOD_ood.score(pred_probs=pred_probs)
    outlier_score = OOD_outlier.score(features=features)

    assert ood_score is not None and outlier_score is not None
    assert np.sum(ood_score) != np.sum(outlier_score)

    #### TESTING FIT SCORE
    OOD_ood_fs = OutOfDistribution()
    ood_score_fs = OOD_ood_fs.fit_score(pred_probs=pred_probs, labels=labels)
    OOD_outlier_fs = OutOfDistribution()
    outlier_score_fs = OOD_outlier_fs.fit_score(features=features)

    assert OOD_ood_fs.confident_thresholds is not None and OOD_ood_fs.knn is None
    assert (OOD_ood_fs.confident_thresholds == OOD_ood.confident_thresholds).all()

    assert OOD_outlier.knn is not None and OOD_outlier.confident_thresholds is None

    assert OOD_ood.params == OOD_ood_fs.params and OOD_outlier.params == OOD_outlier_fs.params
    assert ood_score_fs is not None and outlier_score_fs is not None

    assert np.sum(outlier_score_fs) - np.sum(outlier_score) < 1  # scores are similar
    assert np.sum(ood_score_fs) - np.sum(ood_score) < 1  # scores are similar
