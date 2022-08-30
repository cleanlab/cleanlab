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
from cleanlab import count, outlier
from cleanlab.count import get_confident_thresholds
from cleanlab.outlier import OutOfDistribution
from cleanlab.internal.label_quality_utils import get_normalized_entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression as LogReg


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


def test_class_wrong_info_assert_valid_inputs():
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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_class_wrong_info_fit_ood():
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
        assert "OOD Object needs to be fit on features first." in str(e)
        with pytest.raises(ValueError) as e:
            OOD.score(pred_probs=pred_probs)

    # TESTING: calling scoring with opposite fitting
    OOD_outlier = OutOfDistribution()
    OOD_outlier.fit(features=features)
    try:
        OOD_outlier.score(pred_probs=pred_probs)
    except Exception as e:
        assert "OOD Object needs to be fit on features first." in str(e)
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

    # TESTING: calling ood after fitting but without adjust_pred_probs
    OOD_ood = OutOfDistribution()

    #  This should throw warning since no confident_thresholds calculated
    OOD_ood.fit(pred_probs=pred_probs, params={"adjust_pred_probs": False})
    OOD_ood.score(pred_probs=pred_probs)  # This should be ok since we are not adjusting


def test_class_params_logic():
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

    # Test fitting OOD object without labels and adjust_pred_probs=False
    OOD_ood = OutOfDistribution()
    OOD_ood.fit(pred_probs=data["pred_probs"], labels=None, params={"adjust_pred_probs": False})
    assert OOD_ood.params["adjust_pred_probs"] is False

    # Testing regular fit
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

    #### TESTING PASS IN OTHER KNN
    knn1 = NearestNeighbors(n_neighbors=7, metric="cosine")
    knn2 = NearestNeighbors(n_neighbors=17, metric="cosine")

    OOD_knn1 = OutOfDistribution()
    OOD_knn2 = OutOfDistribution()
    OOD_knn0 = OutOfDistribution()

    scores_knn0 = OOD_knn0.fit_score(features=features)
    scores_knn1 = OOD_knn1.fit_score(features=features, knn=knn1)
    scores_knn2 = OOD_knn2.fit_score(features=features, knn=knn2)

    assert np.sum(scores_knn0) != np.sum(scores_knn1) and np.sum(scores_knn0) != np.sum(scores_knn2)
    assert (
        OOD_knn1.knn.n_neighbors == 7
        and OOD_knn2.knn.n_neighbors == 17
        and OOD_knn0.knn.n_neighbors == 10
    )


def test_get_ood_features_scores():
    X_train = data["X_train"]
    X_test = data["X_test"]

    # Create OOD datapoint
    X_ood = np.array([[999999999.0, 999999999.0]])

    # Add OOD datapoint to X_test
    X_test_with_ood = np.vstack([X_test, X_ood])

    # Fit nearest neighbors on X_train
    knn = NearestNeighbors(n_neighbors=5).fit(X_train)

    # Get KNN distance as outlier score
    k = 5
    knn_distance_to_score, _ = outlier._get_ood_features_scores(
        features=X_test_with_ood, knn=knn, k=k
    )

    # Checking that X_ood has the smallest outlier score among all the datapoints
    assert np.argmin(knn_distance_to_score) == (knn_distance_to_score.shape[0] - 1)

    # Get KNN distance as outlier score without passing k
    # By default k=10 is used or k = n_neighbors when k > n_neighbors extracted from the knn
    knn_distance_to_score, _ = outlier._get_ood_features_scores(features=X_test_with_ood, knn=knn)
    # Checking that X_ood has the smallest outlier score among all the datapoints
    assert np.argmin(knn_distance_to_score) == (knn_distance_to_score.shape[0] - 1)

    # Get KNN distance as outlier score passing k and t > 1
    large_t_knn_distance_to_score, _ = outlier._get_ood_features_scores(
        features=X_test_with_ood, knn=knn, k=k, t=5
    )

    # Checking that X_ood has the smallest outlier score among all the datapoints
    assert np.argmin(large_t_knn_distance_to_score) == (large_t_knn_distance_to_score.shape[0] - 1)

    # Get KNN distance as outlier score passing k and t < 1
    small_t_knn_distance_to_score, _ = outlier._get_ood_features_scores(
        features=X_test_with_ood, knn=knn, k=k, t=0.002
    )

    # Checking that X_ood has the smallest outlier score among all the datapoints
    assert np.argmin(small_t_knn_distance_to_score) == (small_t_knn_distance_to_score.shape[0] - 1)
    assert np.sum(small_t_knn_distance_to_score) >= np.sum(large_t_knn_distance_to_score)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_k_and_model_get_ood_features_scores():
    # Testing using 'None' as model param and correct setting of default k as max_k

    # Create dataset with OOD example
    X = data["X_test"]
    X_ood = np.array([[999999999.0, 999999999.0]])
    X_with_ood = np.vstack([X, X_ood])

    instantiated_k = 10

    # Create NN class object with small instantiated k and fit on data
    knn = NearestNeighbors(n_neighbors=instantiated_k, metric="cosine").fit(X_with_ood)

    avg_knn_distances_default_model, _ = outlier._get_ood_features_scores(
        features=X_with_ood,
        k=instantiated_k,  # this should use default estimator (same as above) and k = instantiated_k
    )

    avg_knn_distances_default_k, knn2 = outlier._get_ood_features_scores(
        features=X_with_ood,  # default k should be set to 10 == instantiated_k
    )
    assert isinstance(knn2, type(knn))

    avg_knn_distances, _ = outlier._get_ood_features_scores(
        features=None,
        knn=knn,
        k=25,  # this should throw user warn, k should be set to instantiated_k
    )

    # Score sums should be equal because the three estimators used have identical params and fit
    assert avg_knn_distances.sum() == avg_knn_distances_default_model.sum()
    assert avg_knn_distances_default_k.sum() == avg_knn_distances.sum()

    avg_knn_distances_large_k, _ = outlier._get_ood_features_scores(
        features=X_with_ood,
        k=25,  # this should use default estimator and k = 25
    )

    avg_knn_distances_tiny_k, _ = outlier._get_ood_features_scores(
        features=None,
        knn=knn,
        k=1,  # this should use knn estimator and k = 1
    )

    avg_knn_distances_tiny_k_default, _ = outlier._get_ood_features_scores(
        features=X_with_ood,
        k=1,  # this should use default estimator and k = 1
    )

    # Score sums should be different because k = user param for estimators and k != 10.
    assert avg_knn_distances_tiny_k.sum() != avg_knn_distances.sum()
    assert avg_knn_distances_large_k.sum() != avg_knn_distances.sum()
    assert avg_knn_distances_tiny_k_default.sum() != avg_knn_distances_default_model.sum()

    # Test that when knn is None ValueError raised if passed in k > len(features)
    try:
        outlier._get_ood_features_scores(
            features=X_with_ood,
            knn=None,
            k=len(X_with_ood) + 1,  # this should throw ValueError, k ! > len(features)
        )
    except Exception as e:
        assert "nearest neighbors" in str(e)
        with pytest.raises(ValueError) as e:
            outlier._get_ood_features_scores(
                features=X_with_ood,
                knn=None,
                k=len(X_with_ood) + 1,  # this should throw ValueError, k ! > len(features)
            )


def test_not_enough_info_get_ood_features_scores():
    # Testing calling function with not enough information to calculate outlier scores
    try:
        outlier._get_ood_features_scores(
            features=None,
            knn=None,  # this should throw TypeError because knn=None and features=None
        )
    except Exception as e:
        assert "Both knn and features arguments" in str(e)
        with pytest.raises(ValueError) as e:
            outlier._get_ood_features_scores(
                features=None,
                knn=None,  # this should throw TypeError because knn=None and features=None
            )


def test_ood_predictions_scores():
    # Create and add OOD datapoint to test set
    X = data["X_test"]
    X_ood = np.array([[999999999.0, 999999999.0]])
    X_with_ood = np.vstack([X, X_ood])

    y = data["true_labels_test"]
    y_with_ood = np.hstack([y, data["true_labels_train"][1]])

    # Fit Logistic Regression model on X_train and estimate pred_probs
    logreg = LogReg(multi_class="auto", solver="lbfgs")
    logreg.fit(data["X_train"], data["true_labels_train"])
    pred_probs = logreg.predict_proba(X_with_ood)

    ### Test non-adjusted OOD score logic
    ood_predictions_scores_entropy, _ = outlier._get_ood_predictions_scores(
        pred_probs=pred_probs,
        adjust_pred_probs=False,
    )

    # adjust pred probs should be False by default
    ood_predictions_scores_least_confidence, _ = outlier._get_ood_predictions_scores(
        pred_probs=pred_probs,
        method="least_confidence",
        adjust_pred_probs=False,
    )

    # check OOD scores calculated correctly
    assert (get_normalized_entropy(pred_probs) == ood_predictions_scores_entropy).all()
    assert (1.0 - pred_probs.max(axis=1) == ood_predictions_scores_least_confidence).all()

    ### Test adjusted OOD score logic
    (
        ood_predictions_scores_adj_entropy,
        confident_thresholds_adj_entropy,
    ) = outlier._get_ood_predictions_scores(
        pred_probs=pred_probs,
        labels=y_with_ood,
        adjust_pred_probs=True,
        method="entropy",
    )

    (
        ood_predictions_scores_adj_least_confidence,
        confident_thresholds_adj_least_confidence,
    ) = outlier._get_ood_predictions_scores(
        pred_probs=pred_probs,
        labels=y_with_ood,
        adjust_pred_probs=True,
        method="least_confidence",
    )

    # test confident thresholds calculated correctly
    confident_thresholds = get_confident_thresholds(
        labels=y_with_ood, pred_probs=pred_probs, multi_label=False
    )

    assert (confident_thresholds == confident_thresholds_adj_entropy).all()
    assert (confident_thresholds_adj_least_confidence == confident_thresholds_adj_entropy).all()

    # check adjusted OOD scores different from non adjust OOD scores
    assert not (ood_predictions_scores_adj_entropy == ood_predictions_scores_entropy).all()
    assert not (
        ood_predictions_scores_adj_least_confidence == ood_predictions_scores_least_confidence
    ).all()

    ### Test pre-calculated confident thresholds logic
    ood_predictions_scores_2, confident_thresholds_2 = outlier._get_ood_predictions_scores(
        pred_probs=pred_probs,
        confident_thresholds=confident_thresholds,
        adjust_pred_probs=True,
    )

    assert (confident_thresholds_2 == confident_thresholds).all()
    assert (ood_predictions_scores_2 == ood_predictions_scores_adj_entropy).all()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_wrong_info_get_ood_predictions_scores():
    # Test calling function with not enough information to calculate ood scores
    try:
        outlier._get_ood_predictions_scores(
            pred_probs=data["pred_probs"],
            labels=None,
            adjust_pred_probs=True,  # this should throw ValueError because knn=None and features=None
        )
    except Exception as e:
        assert "Cannot calculate adjust_pred_probs without labels" in str(e)
        with pytest.raises(ValueError) as e:
            outlier._get_ood_predictions_scores(
                pred_probs=data["pred_probs"],
                labels=None,
                adjust_pred_probs=True,  # this should throw ValueError because knn=None and features=None
            )

    # Test calling function with not enough information to calculate ood scores
    try:
        outlier._get_ood_predictions_scores(
            pred_probs=data["pred_probs"],
            adjust_pred_probs=True,  # this should throw ValueError because knn=None and features=None
        )
    except Exception as e:
        assert "Cannot calculate adjust_pred_probs without labels" in str(e)
        with pytest.raises(ValueError) as e:
            outlier._get_ood_predictions_scores(
                pred_probs=data["pred_probs"],
                adjust_pred_probs=True,  # this should throw ValueError because not enough data provided
            )

    # Test calling function with not a real method
    try:
        outlier._get_ood_predictions_scores(
            pred_probs=data["pred_probs"],
            labels=data["labels"],
            adjust_pred_probs=True,
            method="not_a_real_method",  # this should throw ValueError because method not real method
        )
    except Exception as e:
        assert "not a valid OOD scoring" in str(e)
        with pytest.raises(ValueError) as e:
            outlier._get_ood_predictions_scores(
                pred_probs=data["pred_probs"],
                labels=data["labels"],
                adjust_pred_probs=True,
                method="not_a_real_method",  # this should throw ValueError because method not real method
            )

    # Test calling function with too much information to calculate ood scores
    outlier._get_ood_predictions_scores(
        pred_probs=data["pred_probs"],
        labels=data["labels"],
        adjust_pred_probs=False,  # this should user warning because provided info is not used
    )
