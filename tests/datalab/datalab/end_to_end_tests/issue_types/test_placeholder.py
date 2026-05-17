import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab

SEED = 42


def _build_features_with_placeholder():
    rng = np.random.default_rng(SEED)
    features = rng.uniform(10, 80, size=(200, 2))
    features[::4, 0] = -99
    return features


@pytest.fixture
def lab_with_features():
    features = _build_features_with_placeholder()
    df = pd.DataFrame(features, columns=["age", "income"])
    df["label"] = (rng_labels := np.random.default_rng(SEED)).integers(0, 2, size=len(df))
    lab = Datalab(data=df, label_name="label")
    return lab, features


def test_lab_find_issues_placeholder_numpy(lab_with_features):
    lab, features = lab_with_features
    lab.find_issues(features=features, issue_types={"placeholder": {}})

    placeholder_issues = lab.get_issues("placeholder")
    assert placeholder_issues["is_placeholder_issue"].sum() > 0
    assert placeholder_issues["placeholder_score"].mean() < 1.0

    info = lab.get_info("placeholder")
    assert any(
        any(np.isclose(v, -99) for v in vals) for vals in info["placeholder_by_column"].values()
    )


def test_lab_find_issues_placeholder_dataframe(lab_with_features):
    lab, features = lab_with_features
    df = pd.DataFrame(features, columns=["age", "income"])
    lab.find_issues(features=df, issue_types={"placeholder": {}})

    info = lab.get_info("placeholder")
    assert "age" in info["placeholder_by_column"]


def test_lab_find_issues_placeholder_clean_data():
    rng = np.random.default_rng(SEED)
    features = rng.uniform(10, 80, size=(200, 3))
    df = pd.DataFrame(features, columns=["c1", "c2", "c3"])
    df["label"] = rng.integers(0, 2, size=len(df))
    lab = Datalab(data=df, label_name="label")

    lab.find_issues(features=features, issue_types={"placeholder": {}})
    placeholder_issues = lab.get_issues("placeholder")
    assert not placeholder_issues["is_placeholder_issue"].any()
