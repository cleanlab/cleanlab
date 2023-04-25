from cleanlab.object_detection.rank import (
    get_label_quality_scores,
    issues_from_scores,
    visualize,
    _get_min_pred_prob,
    _softmax,
    _softmin1D,
    _get_valid_score,
    _bbox_xyxy_to_xywh,
    _prune_by_threshold,
    _compute_label_quality_scores,
    _separate_label,
    _separate_prediction,
    _get_overlap_matrix,
    _get_dist_matrix,
)

import numpy as np

import warnings

import pytest

from PIL import Image
import numpy as np

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


warnings.filterwarnings("ignore")

predictions = [
    [
        [],
        [
            [135.011, 235.428, 242.794, 280.324, 0.442],
            [277.164, 370.205, 333.007, 426.35, 0.213],
            [618.572, 345.519, 639.224, 426.451, 0.106],
            [136.026, 249.197, 152.126, 274.018, 0.096],
            [135.011, 247.702, 193.286, 273.123, 0.086],
            [136.58, 248.428, 165.579, 273.118, 0.075],
            [247.92, 341.299, 308.505, 425.982, 0.054],
        ],
        [
            [157.144, 113.92, 173.256, 129.331, 0.742],
            [144.433, 269.503, 172.975, 303.368, 0.69],
            [121.25, 274.145, 144.453, 306.208, 0.526],
            [249.437, 340.538, 300.789, 425.402, 0.442],
            [155.648, 168.818, 182.364, 184.548, 0.391],
            [322.219, 60.97, 337.243, 78.251, 0.23],
            [358.432, 63.371, 382.29, 87.495, 0.162],
            [61.763, 32.829, 123.159, 135.47, 0.12],
            [337.75, 60.75, 356.248, 78.516, 0.107],
            [203.805, 131.003, 258.593, 193.753, 0.096],
            [326.321, 164.405, 358.241, 205.042, 0.09],
            [121.438, 273.073, 138.858, 296.033, 0.079],
            [30.629, 341.96, 99.499, 385.207, 0.077],
            [346.648, 83.467, 369.731, 108.147, 0.069],
            [216.162, 261.51, 256.074, 298.204, 0.068],
            [88.995, 5.224, 139.134, 68.347, 0.067],
            [253.883, 116.387, 293.624, 159.546, 0.065],
            [135.494, 276.395, 148.796, 304.116, 0.059],
        ],
        [
            [387.278, 68.7, 499.833, 345.118, 0.9999],
            [0.0, 261.727, 63.421, 305.305, 0.893],
            [464.847, 87.14, 513.476, 136.868, 0.054],
        ],
        [],
    ],
    [
        [],
        [[241.3, 177.116, 297.747, 228.405, 0.116], [84.284, 188.507, 169.367, 228.18, 0.107]],
        [[209.936, 122.989, 215.953, 133.578, 0.266], [335.776, 55.317, 352.0, 77.911, 0.07]],
        [],
        [],
    ],
    [
        [
            [591.134, 277.14, 638.379, 340.738, 0.304],
            [611.689, 251.713, 639.139, 337.745, 0.235],
            [597.653, 230.408, 635.975, 339.172, 0.068],
        ],
        [[197.521, 227.798, 332.14, 374.322, 0.195]],
        [[346.335, 226.451, 357.505, 247.975, 0.173]],
        [
            [6.485, 163.255, 136.174, 395.704, 0.999],
            [328.483, 173.752, 396.547, 372.454, 0.999],
            [507.475, 172.329, 630.554, 384.995, 0.999],
            [613.713, 245.157, 637.14, 337.998, 0.063],
        ],
        [[338.683, 42.592, 399.161, 106.331, 0.845]],
    ],
    [
        [],
        [
            [164.599, 290.227, 358.904, 494.316, 0.841],
            [303.496, 345.948, 351.585, 402.053, 0.176],
            [0.043, 282.055, 18.413, 308.25, 0.111],
            [307.782, 352.802, 355.269, 459.404, 0.079],
            [309.075, 356.346, 340.737, 411.017, 0.064],
        ],
        [],
        [
            [1.142, 222.285, 82.401, 306.364, 0.94],
            [96.074, 202.809, 121.392, 300.851, 0.106],
            [100.471, 201.552, 128.465, 344.904, 0.058],
            [2.253, 228.026, 41.348, 278.981, 0.056],
        ],
        [],
    ],
    [
        [
            [0.0, 46.954, 36.545, 71.575, 0.734],
            [0.206, 113.128, 23.607, 186.366, 0.508],
            [0.005, 58.389, 7.103, 69.996, 0.258],
            [11.682, 69.438, 160.726, 213.097, 0.164],
            [46.527, 41.948, 70.301, 56.561, 0.144],
            [2.258, 353.964, 637.883, 459.634, 0.106],
            [23.833, 58.832, 36.728, 70.771, 0.105],
        ],
        [[9.181, 112.75, 187.607, 372.173, 0.191], [52.113, 222.224, 190.611, 374.671, 0.059]],
        [],
        [
            [561.302, 269.395, 600.368, 344.598, 0.994],
            [253.825, 108.051, 272.708, 171.272, 0.947],
            [259.138, 108.331, 273.064, 132.699, 0.145],
            [260.174, 108.24, 273.084, 155.805, 0.056],
        ],
        [],
    ],
]
labels = [
    {
        "bboxes": [
            [388.6600036621094, 69.91999816894531, 498.07000732421875, 347.5400085449219],
            [0.0, 262.80999755859375, 62.15999984741211, 299.5799865722656],
            [119.4000015258789, 272.510009765625, 144.22000122070312, 306.760009765625],
            [141.47000122070312, 267.9100036621094, 173.66000366210938, 303.7699890136719],
        ],
        "labels": [3, 3, 2, 2],
        "seg_map": "000000397133.png",
    },
    {
        "bboxes": [
            [26.5, 215.25, 88.0, 229.75],
            [116.5, 189.57000732421875, 166.5, 215.07000732421875],
            [241.9499969482422, 180.4199981689453, 293.32000732421875, 225.82000732421875],
        ],
        "labels": [1, 1, 1],
        "seg_map": "000000037777.png",
    },
    {
        "bboxes": [
            [326.2799987792969, 174.55999755859375, 397.5199890136719, 371.80999755859375],
            [9.789999961853027, 167.05999755859375, 131.72999572753906, 393.510009765625],
            [510.44000244140625, 171.27000427246094, 634.0999755859375, 387.0299987792969],
            [345.1300048828125, 226.41000366210938, 356.19000244140625, 248.5500030517578],
            [337.05999755859375, 44.11000061035156, 398.4200134277344, 101.27999877929688],
        ],
        "labels": [3, 3, 3, 2, 4],
        "seg_map": "000000252219.png",
    },
    {
        "bboxes": [[167.3800048828125, 293.55999755859375, 354.0799865722656, 492.05999755859375]],
        "labels": [1],
        "seg_map": "000000491497.png",
    },
    {
        "bboxes": [
            [567.8200073242188, 273.1000061035156, 599.2000122070312, 347.2099914550781],
            [251.19000244140625, 106.41999816894531, 274.510009765625, 168.13999938964844],
        ],
        "labels": [3, 3],
        "seg_map": "000000348881.png",
    },
]


def make_numpy(labels, predictions):
    np_labels = []
    for ann in labels:
        np_labels.append(
            {
                "bboxes": np.array(ann["bboxes"]),
                "labels": np.array(ann["labels"]),
                "seg_map": ann["seg_map"],
            }
        )

    np_predictions = predictions[:]
    np_predictions = [np.array(pred, dtype=object) for pred in np_predictions]
    for i in range(len(np_predictions)):
        for j in range(len(np_predictions[i])):
            if len(np_predictions[i][j]) == 0:
                np_predictions[i][j] = np.zeros((0, 5))
            else:
                np_predictions[i][j] = np.array(np_predictions[i][j])
    return np_labels, np_predictions


labels, predictions = make_numpy(labels, predictions)


def test_get_label_quality_scores():
    scores = get_label_quality_scores(labels, predictions)
    assert len(scores) == len(labels)
    assert (scores <= 1.0).all()
    assert len(scores.shape) == 1


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
    assert min == 0.054


def test_get_valid_score():
    score = _get_valid_score([])
    assert score == 1.0

    score_larger = _get_valid_score([0.8, 0.7, 0.6])
    score_smaller = _get_valid_score([0.8, 0.7, 0.6], temperature=0.2)
    assert score_smaller < score_larger


def test_softmin1D():
    small_val = 0.004
    assert _softmin1D([small_val]) == small_val


def test_softmax():
    small_val = 0.004
    assert _softmax(np.array([small_val])) == 1.0


def test_bbox_xyxy_to_xywh():
    box_coords = _bbox_xyxy_to_xywh([5, 4, 2, 5, 0.86])
    assert box_coords is None
    box_coords = _bbox_xyxy_to_xywh([5, 4, 2, 5])
    assert box_coords is not None


@pytest.mark.filterwarnings("ignore::UserWarning")  # Should be 2 warnings (first two calls)
def test_prune_by_threshold():
    pruned_predictions = _prune_by_threshold(predictions, 1.0)
    print(pruned_predictions)
    for image_pred in pruned_predictions:
        for class_pred in image_pred:
            assert class_pred.shape[0] == 0

    pruned_predictions = _prune_by_threshold(predictions, 0.9999)

    num_boxes_not_pruned = 0
    for image_pred in pruned_predictions:
        for class_pred in image_pred:
            if class_pred.shape[0] > 0:
                num_boxes_not_pruned += 1
    assert num_boxes_not_pruned == 1

    pruned_predictions = _prune_by_threshold(predictions, 0.0)
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
    scores_with_threshold = _compute_label_quality_scores(labels, predictions, threshold=0.9)
    assert np.sum(scores) != np.sum(scores_with_threshold)

    min_pred_prob = _get_min_pred_prob(predictions)
    scores_with_min_threshold = _compute_label_quality_scores(
        labels, predictions, threshold=min_pred_prob
    )
    assert (scores == scores_with_min_threshold).all()


@pytest.mark.usefixtures("generate_single_image_file")
def test_visualize(monkeypatch, generate_single_image_file):
    monkeypatch.setattr(plt, "show", lambda: None)
    visualize(generate_single_image_file, labels[0], predictions[0])
    visualize(
        generate_single_image_file,
        labels[0],
        predictions[0],
        prediction_threshold=0.99,
        given_label_overlay=False,
    )
