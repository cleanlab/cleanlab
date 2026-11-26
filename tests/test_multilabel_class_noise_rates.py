import numpy as np
import pandas as pd
import pytest

from cleanlab.internal.multilabel_utils import onehot2int
from cleanlab.multilabel_classification.dataset import (
    common_multilabel_issues,
    multilabel_health_summary,
    rank_classes_by_multilabel_quality,
)


def imbalanced_data(clean_negative_rows=0):
    given = np.zeros((100 + clean_negative_rows, 3), dtype=int)
    given[:10, 0] = 1
    given[10:80, 1] = 1
    probabilities = np.where(given, 0.99, 0.01)
    probabilities[:5, 0] = 0.01
    probabilities[96:100, 0] = 0.99
    probabilities[10:17, 1] = 0.01
    return onehot2int(given), probabilities


@pytest.mark.parametrize("class_names", [None, ["rare", "common", "absent"]])
def test_class_quality_is_conditioned_on_examples_with_that_label(class_names):
    labels, probabilities = imbalanced_data()
    ranked = rank_classes_by_multilabel_quality(labels, probabilities, class_names=class_names)
    assert ranked["Class Index"].tolist() == [0, 1, 2]
    indexed = ranked.set_index("Class Index")
    np.testing.assert_array_equal(indexed["Label Issues"], [5, 7, 0])
    np.testing.assert_array_equal(indexed["Inverse Label Issues"], [4, 0, 0])
    np.testing.assert_allclose(indexed["Label Noise"], [5 / 10, 7 / 70, 0])
    np.testing.assert_allclose(indexed["Label Quality Score"], [0.5, 0.9, 1])
    # Estimated true class membership adds four missing annotations and removes five wrong ones.
    np.testing.assert_allclose(indexed["Inverse Label Noise"], [4 / (10 - 5 + 4), 0, 0])
    if class_names is not None:
        assert ranked["Class Name"].tolist() == class_names


def test_clean_examples_without_a_class_do_not_dilute_its_label_noise():
    labels, probabilities = imbalanced_data()
    padded_labels, padded_probabilities = imbalanced_data(clean_negative_rows=100)
    original = rank_classes_by_multilabel_quality(labels, probabilities).set_index("Class Index")
    padded = rank_classes_by_multilabel_quality(padded_labels, padded_probabilities).set_index(
        "Class Index"
    )
    pd.testing.assert_frame_equal(original, padded, check_exact=True)


def test_counts_are_not_lost_in_a_probability_round_trip():
    given = np.zeros((49, 2), dtype=int)
    given[:5, 0] = 1
    given[10:30, 1] = 1
    probabilities = np.where(given, 0.99, 0.01)
    probabilities[0, 0] = 0.01
    labels = onehot2int(given)
    common = common_multilabel_issues(labels, probabilities)
    assert (
        common.loc[(common["Class Index"] == 0) & common["In Given Label"], "Num Examples"].item()
        == 1
    )
    ranked = rank_classes_by_multilabel_quality(labels, probabilities).set_index("Class Index")
    assert ranked.loc[0, "Label Issues"] == 1
    assert ranked.loc[0, "Label Noise"] == pytest.approx(1 / 5)


def test_health_summary_uses_the_same_conditional_class_rates():
    labels, probabilities = imbalanced_data()
    summary = multilabel_health_summary(labels, probabilities, verbose=False)
    ranked = summary["classes_by_multilabel_quality"].set_index("Class Index")
    assert ranked.loc[0, "Label Quality Score"] == 0.5
    assert ranked.loc[1, "Label Quality Score"] == 0.9
    assert summary["overall_multilabel_health_score"] == 0.84
    # The separate table intentionally reports dataset-wide frequencies.
    common = summary["common_multilabel_issues"]
    assert (
        common.loc[
            (common["Class Index"] == 0) & common["In Given Label"], "Issue Probability"
        ].item()
        == 0.05
    )


def test_missing_class_and_clean_classes_keep_finite_scores():
    given = np.zeros((30, 3), dtype=int)
    given[:10, 0] = 1
    given[10:20, 1] = 1
    ranked = rank_classes_by_multilabel_quality(onehot2int(given), np.where(given, 0.99, 0.01))
    np.testing.assert_array_equal(ranked["Label Issues"], 0)
    np.testing.assert_array_equal(ranked["Inverse Label Issues"], 0)
    np.testing.assert_array_equal(ranked["Label Noise"], 0)
    np.testing.assert_array_equal(ranked["Inverse Label Noise"], 0)
    np.testing.assert_array_equal(ranked["Label Quality Score"], 1)
