from cleanlab.object_detection.rank import (
    get_label_quality_scores,
    issues_from_scores,
)

import numpy as np

import warnings
import pickle

warnings.filterwarnings("ignore")

results = pickle.load(open("tests/object_detection_test_data/results.pkl", "rb"))
dataset = pickle.load(open("tests/object_detection_test_data/dataset.pkl", "rb"))
test_length = 5


def test_get_label_quality_scores():
    scores = get_label_quality_scores(dataset[:test_length], results[:test_length])
    assert len(scores) == len(dataset[:test_length])
    assert (scores < 1.0).all()
    assert len(scores.shape) == 1


def test_issues_from_scores():
    scores = get_label_quality_scores(dataset[:test_length], results[:test_length])
    real_issue_from_scores = issues_from_scores(scores, threshold=1.0)
    print(np.argmin(scores))
    print(real_issue_from_scores[0])
    print(real_issue_from_scores)
    assert len(real_issue_from_scores) == len(scores)
    assert np.argmin(scores) == real_issue_from_scores[0]

    fake_scores = np.array([0.2, 0.4, 0.6, 0.1])
    fake_threshold = 0.3
    fake_issue_from_scores = issues_from_scores(fake_scores, threshold=fake_threshold)
    assert (fake_issue_from_scores == np.array([3, 0])).all()
