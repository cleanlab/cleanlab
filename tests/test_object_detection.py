import os

from cleanlab.internal.object_detection_utils import (
    softmin1d,
    softmax,
    bbox_xyxy_to_xywh,
)

from cleanlab.object_detection.rank import (
    get_label_quality_scores,
    issues_from_scores,
    _get_min_pred_prob,
    _get_valid_score,
    _prune_by_threshold,
    _compute_label_quality_scores,
    _separate_label,
    _separate_prediction,
    _get_overlap_matrix,
    _get_dist_matrix,
    _get_valid_inputs_for_compute_scores,
    _get_valid_inputs_for_compute_scores_per_image,
    compute_overlooked_box_scores,
    compute_badloc_box_scores,
    compute_swap_box_scores,
    _get_prediction_type,
    _get_valid_subtype_score_params,
    _get_aggregation_weights,
)

from cleanlab.object_detection.filter import (
    find_label_issues,
    _find_label_issues_per_box,
    _pool_box_scores_per_image,
    _find_label_issues,
)

from cleanlab.object_detection.summary import (
    visualize,
)
from cleanlab.internal.constants import (
    ALPHA,
    LOW_PROBABILITY_THRESHOLD,
    HIGH_PROBABILITY_THRESHOLD,
    OVERLOOKED_THRESHOLD,
    BADLOC_THRESHOLD,
    SWAP_THRESHOLD,
    TEMPERATURE,
    CUSTOM_SCORE_WEIGHT_OVERLOOKED,
    CUSTOM_SCORE_WEIGHT_SWAP,
    CUSTOM_SCORE_WEIGHT_BADLOC,
)

import numpy as np

np.random.seed(0)

import warnings

import pytest

from PIL import Image
import numpy as np
import copy

# to suppress plt.show()
import matplotlib.pyplot as plt


def generate_image(arr=None):
    """Generates single image of randomly colored pixels"""
    if arr is None:
        arr = np.random.randint(low=0, high=256, size=(300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img


@pytest.fixture(scope="session")
def generate_single_image_file(tmpdir_factory, img_name="img.png", arr=None):
    """Generates a single temporary image for testing"""
    img = generate_image(arr)
    fn = tmpdir_factory.mktemp("data").join(img_name)
    img.save(str(fn))
    return str(fn)


@pytest.fixture(scope="session")
def generate_n_image_files(tmpdir_factory, n=5):
    """Generates n temporary images for testing and returns dir of images"""
    filename_list = []
    tmp_image_dir = tmpdir_factory.mktemp("data")
    for i in range(n):
        img = generate_image()
        img_name = f"{i}.png"
        fn = tmp_image_dir.join(img_name)
        img.save(str(fn))
        filename_list.append(str(fn))
    return str(tmp_image_dir)


def generate_predictions(
    num_predictions, annotations, num_classes=5, max_boxes=6, image_size=300, is_issue=False
):
    """Generates num_predictions number of predictions based on passed in hyperparameters in same format as expected by find_label_issues and get_label_quality_scores"""

    predictions = []
    if isinstance(is_issue, int):
        is_issue = [is_issue] * num_predictions
    for i in range(num_predictions):
        issue = is_issue[i]
        annotation = annotations[i] if i < len(annotations) else None
        prediction = generate_prediction(annotation, num_classes, image_size, max_boxes, issue)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def generate_prediction(annotation, num_classes, image_size, max_boxes, issue):
    """Generates a single prediction based on passed in hyperparameters in same format as expected by find_label_issues and get_label_quality_scores"""

    prediction = [[] for _ in range(num_classes)]
    if annotation is None and issue is False:
        return
    else:
        if issue is False:
            for label, bboox in zip(annotation["labels"], annotation["bboxes"]):
                rand_probability = np.random.randint(low=96, high=100) / 100
                prediction[label].append(list(bboox) + [rand_probability])
        else:
            num_predictions = np.random.randint(low=1, high=max_boxes + 1)
            rand_labels = generate_labels(num_classes, num_predictions)
            for label in rand_labels:
                rand_bbox = generate_bbox(image_size)
                rand_probability = np.random.randint(low=96, high=100) / 100
                prediction[label].append(list(rand_bbox) + [rand_probability])
        prediction = [
            np.array(p) if len(p) > 0 else np.empty(shape=[0, 5], dtype=np.float32)
            for p in prediction
        ]
    return np.array(prediction, dtype=object)


def generate_annotations(num_annotations, num_classes=5, max_boxes=5, image_size=300):
    """Generates num_annotations number of annotations based on passed in hyperparameters in same format as expected by find_label_issues and get_label_quality_scores"""

    annotations = []
    for i in range(num_annotations):
        annotations.append(generate_annotation(num_classes, image_size, max_boxes))
    return annotations


def generate_annotation(num_classes, image_size, max_boxes):
    """Generates a single annotation based on passed in hyperparameters in same format as expected by find_label_issues and get_label_quality_scores"""

    num_boxes = np.random.randint(low=1, high=max_boxes)
    bboxes = np.array([generate_bbox(image_size) for _ in range(num_boxes)])
    labels = generate_labels(num_classes, num_boxes)
    annotation = {"bboxes": bboxes, "labels": labels}
    return annotation


def generate_labels(num_classes, num_boxes):
    """Generates num_boxes number of labels with possible values [0-num_classes)"""
    return np.random.choice(num_classes, num_boxes)


def generate_bbox(image_size):
    """Generates a single bounding box x1,y1,x2,y2 with coordinates lower than image_size"""
    x2 = np.random.randint(low=2, high=image_size - 1)
    y2 = np.random.randint(low=2, high=image_size - 1)
    x_shift = np.random.randint(low=1, high=x2)
    y_shift = np.random.randint(low=1, high=y2)
    x1 = x2 - x_shift
    y1 = y2 - y_shift
    return [x1, y1, x2, y2]


warnings.filterwarnings("ignore")

good_labels = generate_annotations(5, num_classes=10, max_boxes=10)
good_predictions = generate_predictions(
    5, good_labels, num_classes=10, max_boxes=12, is_issue=False
)

bad_labels = generate_annotations(5, num_classes=10, max_boxes=10)
bad_predictions = generate_predictions(5, bad_labels, num_classes=10, max_boxes=12, is_issue=True)

labels = good_labels + bad_labels  # 10 labels
predictions = (
    good_predictions + bad_predictions
)  # 10 predictions, [:5] is perfect predictions, [5:] is bad predictions


def test_get_label_quality_scores():
    scores = get_label_quality_scores(labels, predictions)
    assert len(scores) == len(labels)
    assert (scores <= 1.0).all()
    assert len(scores.shape) == 1
    assert (scores[:5] > 0.9).all()  # perfect annotations get high scores
    assert (scores[5:] < 0.7).all()  # label issues get low scores


@pytest.mark.parametrize(
    "agg_weights",
    [
        {"overlooked": 1.0, "swap": 0.0, "badloc": 0.0},
        {"overlooked": 0.0, "swap": 1.0, "badloc": 0.0},
        {"overlooked": 0.0, "swap": 0.0, "badloc": 1.0},
    ],
)
def test_get_label_quality_scores_custom_weights(agg_weights):
    scores = get_label_quality_scores(labels, predictions, aggregation_weights=agg_weights)
    assert (scores[:5] > 0.8).all()  # perfect annotations get high scores

    if agg_weights["swap"] == 1.0:
        assert (scores[5:][scores[5:] != 1.0] < 0.8).any()  # swapped label issues get low scores
    elif agg_weights["overlooked"] == 1.0:
        assert (scores[5:][scores[5:] != 1.0] < 0.7).all()  # overlooked label issues get low scores
    elif agg_weights["badloc"] == 1.0:
        assert (scores[5:][scores[5:] != 1.0] < 0.7).all()  # label issues get low scores


def test_issues_from_scores():
    scores = get_label_quality_scores(labels, predictions)
    real_issue_from_scores = issues_from_scores(scores, threshold=1.0)
    assert len(real_issue_from_scores) == len(scores)
    assert np.argmin(scores) == real_issue_from_scores[0]

    fake_scores = np.array([0.2, 0.4, 0.6, 0.1])
    fake_threshold = 0.3
    fake_issue_from_scores = issues_from_scores(fake_scores, threshold=fake_threshold)
    assert (fake_issue_from_scores == np.array([3, 0])).all()


def test_get_min_pred_prob():
    min = _get_min_pred_prob(predictions)
    assert min == 0.96


def test_get_valid_score():
    score = _get_valid_score(np.array([]), temperature=0.99)
    assert score == 1.0

    score_larger = _get_valid_score(np.array([0.8, 0.7, 0.6]), temperature=0.99)
    score_smaller = _get_valid_score(np.array([0.8, 0.7, 0.6]), temperature=0.2)
    assert score_smaller < score_larger


def test_get_valid_subtype_score_params():
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(None, None, None, None)
    assert alpha == ALPHA
    assert low_probability_threshold == LOW_PROBABILITY_THRESHOLD
    assert high_probability_threshold == HIGH_PROBABILITY_THRESHOLD
    assert temperature == TEMPERATURE


def test_get_aggregation_weights():
    correct_aggregation_weights = {
        "overlooked": CUSTOM_SCORE_WEIGHT_OVERLOOKED,
        "swap": CUSTOM_SCORE_WEIGHT_SWAP,
        "badloc": CUSTOM_SCORE_WEIGHT_BADLOC,
    }
    weights = _get_aggregation_weights(None)
    assert weights == correct_aggregation_weights

    with pytest.raises(ValueError) as e:
        _get_aggregation_weights(
            {
                "overlooked": -1.0,
                "swap": CUSTOM_SCORE_WEIGHT_SWAP,
                "badloc": CUSTOM_SCORE_WEIGHT_BADLOC,
            }
        )

    with pytest.raises(ValueError) as e:
        _get_aggregation_weights(
            {
                "overlooked": CUSTOM_SCORE_WEIGHT_OVERLOOKED,
                "swap": 1.2,
                "badloc": CUSTOM_SCORE_WEIGHT_BADLOC,
            }
        )


def test_softmin1d():
    small_val = 0.004
    assert softmin1d([small_val]) == small_val


def test_softmax():
    small_val = 0.004
    assert softmax(np.array([small_val])) == 1.0


def test_bbox_xyxy_to_xywh():
    box_coords = bbox_xyxy_to_xywh([5, 4, 2, 5, 0.86])
    assert box_coords is None
    box_coords = bbox_xyxy_to_xywh([5, 4, 2, 5])
    assert box_coords is not None


@pytest.mark.filterwarnings("ignore::UserWarning")  # Should be 2 warnings (first two calls)
def test_prune_by_threshold():
    pruned_predictions = _prune_by_threshold(predictions, 1.0)
    print(pruned_predictions)
    for image_pred in pruned_predictions:
        for class_pred in image_pred:
            assert class_pred.shape[0] == 0

    pruned_predictions = _prune_by_threshold(predictions, 0.6)

    num_boxes_not_pruned = 0
    for image_pred in pruned_predictions:
        for class_pred in image_pred:
            if class_pred.shape[0] > 0:
                num_boxes_not_pruned += 1
    assert num_boxes_not_pruned == 44

    pruned_predictions = _prune_by_threshold(predictions, 0.5)
    for im0, im1 in zip(pruned_predictions, predictions):
        for cl0, cl1 in zip(im0, im1):
            assert (cl0 == cl1).all()


def test_similarity_matrix():
    ALPHA = 0.99
    lab_bboxes, lab_labels = _separate_label(labels[0])
    det_bboxes, det_labels, det_label_prob = _separate_prediction(predictions[0])

    iou_matrix = _get_overlap_matrix(lab_bboxes, det_bboxes)
    dist_matrix = 1 - _get_dist_matrix(lab_bboxes, det_bboxes)

    similarity_matrix = iou_matrix * ALPHA + (1 - ALPHA) * (1 - dist_matrix)
    assert (similarity_matrix.flatten() >= 0).all() and (similarity_matrix.flatten() <= 1).all()


def test_compute_label_quality_scores():
    scores = _compute_label_quality_scores(labels, predictions)
    scores_with_threshold = _compute_label_quality_scores(labels, predictions, threshold=0.99)
    assert np.sum(scores) != np.sum(scores_with_threshold)

    min_pred_prob = _get_min_pred_prob(predictions)
    scores_with_min_threshold = _compute_label_quality_scores(
        labels, predictions, threshold=min_pred_prob
    )
    assert (scores == scores_with_min_threshold).all()


def test_overlooked_score_shifts_in_correct_direction():
    perfect_label = labels[0]
    bad_label = copy.deepcopy(labels[0])
    worst_label = copy.deepcopy(labels[0])

    print(predictions[0])
    print(bad_label)
    bad_label["bboxes"] = np.delete(bad_label["bboxes"], 2, axis=0)  # 0.79 pred_probs
    worst_label["bboxes"] = np.delete(worst_label["bboxes"], -1, axis=0)  # 0.84 pred_probs

    bad_label["labels"] = np.delete(bad_label["labels"], 2)
    worst_label["labels"] = np.delete(worst_label["labels"], -1)

    scores = _compute_label_quality_scores(
        [perfect_label, bad_label, worst_label], [predictions[0], predictions[0], predictions[0]]
    )

    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_badloc_score_shifts_in_correct_direction():
    perfect_label = labels[0]
    bad_label = copy.deepcopy(labels[0])
    worst_label = copy.deepcopy(labels[0])

    bad_label["bboxes"][0] = bad_label["bboxes"][0] - 20
    worst_label["bboxes"][0] = worst_label["bboxes"][0] - 100

    scores = _compute_label_quality_scores(
        [perfect_label, bad_label, worst_label], [predictions[0], predictions[0], predictions[0]]
    )
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_swap_score_shifts_in_correct_direction():
    perfect_label = labels[0]
    bad_label = copy.deepcopy(labels[0])
    worst_label = copy.deepcopy(labels[0])

    bad_label["bboxes"][0] = bad_label["bboxes"][0] - 20
    bad_label["labels"][0] = np.random.choice([i for i in range(10) if i != bad_label["labels"][0]])
    worst_label["bboxes"][0] = worst_label["bboxes"][0] - 100
    worst_label["labels"][0] = np.random.choice(
        [i for i in range(10) if i != bad_label["labels"][0]]
    )

    scores = _compute_label_quality_scores(
        [perfect_label, bad_label, worst_label], [predictions[0], predictions[0], predictions[0]]
    )
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_find_label_issues():
    auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)
    test_inputs = _get_valid_inputs_for_compute_scores_per_image(
        alpha=ALPHA, label=labels[0], prediction=predictions[0]
    )

    assert (test_inputs["pred_label_probs"] == auxiliary_inputs[0]["pred_label_probs"]).all()

    overlooked_scores_per_box = compute_overlooked_box_scores(
        alpha=ALPHA,
        high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
        auxiliary_inputs=auxiliary_inputs,
    )

    overlooked_scores_no_auxillary_inputs = compute_overlooked_box_scores(
        alpha=ALPHA,
        high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
        labels=labels,
        predictions=predictions,
    )

    for score, no_auxiliary_inputs_score in zip(
        overlooked_scores_per_box, overlooked_scores_no_auxillary_inputs
    ):
        assert (
            score[~np.isnan(score)]
            == no_auxiliary_inputs_score[~np.isnan(no_auxiliary_inputs_score)]
        ).all()

    overlooked_issues_per_box = _find_label_issues_per_box(
        overlooked_scores_per_box, OVERLOOKED_THRESHOLD
    )
    overlooked_issues_per_image = _pool_box_scores_per_image(overlooked_issues_per_box)
    overlooked_issues = np.sum(overlooked_issues_per_image)
    assert np.sum(overlooked_issues_per_image[5:]) == 5  # check bad labels were detected correctly
    assert overlooked_issues == 5

    badloc_scores_per_box = compute_badloc_box_scores(
        alpha=ALPHA,
        low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
        auxiliary_inputs=auxiliary_inputs,
    )

    badloc_scores_no_auxillary_inputs = compute_badloc_box_scores(
        alpha=ALPHA,
        low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
        labels=labels,
        predictions=predictions,
    )

    for score, no_auxiliary_inputs_score in zip(
        badloc_scores_per_box, badloc_scores_no_auxillary_inputs
    ):
        assert (score == no_auxiliary_inputs_score).all()

    badloc_issues_per_box = _find_label_issues_per_box(badloc_scores_per_box, BADLOC_THRESHOLD)
    badloc_issues_per_image = _pool_box_scores_per_image(badloc_issues_per_box)
    badloc_issues = np.sum(badloc_issues_per_image)
    assert np.sum(badloc_issues_per_image[5:]) == 4  # check bad labels were detected correctly
    assert badloc_issues == 4

    swap_scores_per_box = compute_swap_box_scores(
        alpha=ALPHA,
        high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
        auxiliary_inputs=auxiliary_inputs,
    )

    swap_scores_no_auxillary_inputs = compute_swap_box_scores(
        alpha=ALPHA,
        high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
        labels=labels,
        predictions=predictions,
    )

    for score, no_auxiliary_inputs_score in zip(
        swap_scores_per_box, swap_scores_no_auxillary_inputs
    ):
        assert (score == no_auxiliary_inputs_score).all()

    swap_issues_per_box = _find_label_issues_per_box(swap_scores_per_box, SWAP_THRESHOLD)
    swap_issues_per_image = _pool_box_scores_per_image(swap_issues_per_box)
    swap_issues = np.sum(swap_issues_per_image)
    assert np.sum(swap_scores_per_box[2]) > np.sum(swap_scores_per_box[7])
    assert swap_issues == 0

    label_issues = find_label_issues(labels, predictions)
    assert np.sum(label_issues) == np.sum(
        (swap_issues_per_image + badloc_issues_per_image + overlooked_issues_per_image) > 0
    )
    assert np.sum(label_issues[5:]) == 5  # check bad labels were detected correctly

    swap_issues_per_box = _find_label_issues_per_box(swap_scores_per_box, 0.7)
    swap_issues_per_image = _pool_box_scores_per_image(swap_issues_per_box)
    swap_issues = np.sum(swap_issues_per_image)
    assert swap_issues == 1
    assert np.sum(swap_issues_per_image[5:]) == 1  # check bad labels were detected correctly


def test_separate_prediction():
    pred_bboxes = np.array(
        [
            np.array(list(generate_bbox(300)) + [0.97]),
            np.empty(shape=[0, 5], dtype=np.float32),
            np.array(list(generate_bbox(300)) + [0.94]),
        ],
        dtype=object,
    )
    pred_labels = np.array([0, 2])
    pred_probs = np.array([[0.98, 0.01, 0.01], [0.02, 0.02, 0.98]])
    all_pred_prediction = np.array([pred_bboxes, pred_labels, pred_probs], dtype=object)
    prediction_type = _get_prediction_type(all_pred_prediction)
    assert prediction_type == "all_pred"

    boxes, labels, pred_probs = _separate_prediction(
        all_pred_prediction, prediction_type=prediction_type
    )
    assert len(labels) == len(pred_probs)


def test_return_issues_ranked_by_scores():
    label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)
    assert (
        len(set([5, 6, 7, 8, 9]).intersection(label_issue_idx[:5])) == 5
    )  # lower scores for bad examples
    assert len(label_issue_idx) == 5  # no good example index returned


def test_bad_input_find_label_issues_internal():
    bad_label_issues = _find_label_issues(labels, predictions, scoring_method="bad_method")
    assert (bad_label_issues == -1).all()


def test_find_label_issues_per_box():
    scores_per_box = [np.array([0.2, 0.3]), np.array([]), np.array([0.9, 0.5, 0.9, 0.51])]
    issues_per_box = _find_label_issues_per_box(scores_per_box, threshold=0.5)
    assert issues_per_box[1] == np.array([False])
    assert (issues_per_box[0] == np.array([True, True])).all()
    assert (issues_per_box[2] == np.array([False, True, False, False])).all()


@pytest.mark.usefixtures("generate_single_image_file")
def test_visualize(monkeypatch, generate_single_image_file):
    monkeypatch.setattr(plt, "show", lambda: None)

    arr = np.random.randint(low=0, high=256, size=(300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    visualize(img)

    visualize(img, save_path="./fake_path.pdf")
    assert os.path.exists("./fake_path.pdf")

    visualize(img, save_path="./fake_path_no_ext")
    assert os.path.exists("./fake_path_no_ext.png")

    visualize(img, save_path="./fake_path.ps")
    assert os.path.exists("./fake_path.ps")

    visualize(img, save_path="./fake.path.pdf")
    assert os.path.exists("./fake.path.pdf")

    visualize(generate_single_image_file, label=labels[0], prediction=predictions[0])
    visualize(generate_single_image_file, label=None, prediction=predictions[0])
    visualize(generate_single_image_file, label=labels[0], prediction=None)
    visualize(generate_single_image_file, label=None, prediction=None)

    visualize(generate_single_image_file, label=None, prediction=predictions[0], overlay=False)
    visualize(generate_single_image_file, label=labels[0], prediction=None, overlay=False)
    visualize(generate_single_image_file, label=None, prediction=None, overlay=False)

    visualize(
        generate_single_image_file,
        label=labels[0],
        prediction=predictions[0],
        prediction_threshold=0.99,
        overlay=False,
    )

    visualize(
        generate_single_image_file,
        label=labels[0],
        prediction=predictions[0],
        prediction_threshold=0.99,
        class_names={
            "0": "car",
            "1": "chair",
            "2": "cup",
            "3": "person",
            "4": "traffic light",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
        },
        overlay=False,
    )
