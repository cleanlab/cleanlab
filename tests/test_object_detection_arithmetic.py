import numpy as np
import pytest

from cleanlab.internal.object_detection_utils import (
    bbox_xyxy_to_xywh,
    calculate_bounding_box_areas,
)
from cleanlab.object_detection.rank import (
    _euc_dis,
    _get_overlap,
    _get_overlap_matrix,
    get_label_quality_scores,
)


@pytest.mark.parametrize(
    "dtype,side",
    [
        (np.uint8, 200),
        (np.uint16, 600),
        (np.int16, 600),
        (np.int32, 100000),
        (np.int64, 4000000000),
        (np.float16, 600),
        (np.float32, 600),
        (np.float64, 600),
    ],
)
def test_box_overlaps_and_areas_do_not_overflow_the_coordinate_dtype(dtype, side):
    boxes = np.array([[0, 0, side, side], [0, 0, side // 2, side]], dtype=dtype)
    original = boxes.copy()

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        overlaps = _get_overlap_matrix(boxes, boxes)
        areas = calculate_bounding_box_areas(boxes)

    np.testing.assert_allclose(overlaps, [[1, 0.5], [0.5, 1]], rtol=1e-12)
    np.testing.assert_allclose(areas, [side * side, side * side / 2], rtol=1e-12)
    np.testing.assert_array_equal(boxes, original)


@pytest.mark.parametrize(
    "dtype,left,step,width",
    [(np.uint8, 120, 4, 10), (np.uint16, 32000, 1000, 1000), (np.int16, 15000, 1000, 1000)],
)
def test_center_similarity_does_not_wrap_at_the_integer_boundary(dtype, left, step, width):
    first = np.array([left, 0, left + width, 20], dtype=dtype)
    second = np.array([left + step, 0, left + step + width, 20], dtype=dtype)

    with np.errstate(over="raise", invalid="raise"):
        similarity = _euc_dis(first, second)

    np.testing.assert_allclose(similarity, np.exp(-0.1 * step), rtol=1e-12, atol=0)


def test_integer_box_arithmetic_preserves_small_dimensions_at_large_offsets():
    offset = 2**53
    boxes = np.array([[offset, 0, offset + 1, 1], [offset + 1, 0, offset + 2, 1]])

    np.testing.assert_array_equal(_get_overlap_matrix(boxes, boxes), np.eye(2))
    np.testing.assert_array_equal(calculate_bounding_box_areas(boxes), [1, 1])
    assert bbox_xyxy_to_xywh(boxes[0]) == [offset, 0, 1, 1]


def test_mixed_python_scalars_preserve_integer_coordinates_without_float_coercion():
    offset = 2**53
    box = [offset, 0.0, offset + 1, 1.0]

    assert bbox_xyxy_to_xywh(box) == [offset, 0.0, 1, 1.0]
    assert _get_overlap(box, box) == 1.0


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32])
def test_box_width_conversion_does_not_overflow_signed_coordinates(dtype):
    box = np.array([-30000, -20000, 30000, 20000], dtype=dtype)

    with np.errstate(over="raise", invalid="raise"):
        converted = bbox_xyxy_to_xywh(box)

    assert converted == [-30000, -20000, 60000, 40000]
    np.testing.assert_allclose(calculate_bounding_box_areas(box[None, :]), [2400000000])


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float16, np.float64])
def test_perfect_object_labels_have_the_same_score_for_integer_and_float_boxes(dtype):
    boxes = np.array([[0, 0, 600, 600], [700, 0, 1300, 600]], dtype=dtype)
    labels = [{"bboxes": boxes, "labels": np.array([0, 1])}]
    predictions = [[np.array([[0, 0, 600, 600, 0.99]]), np.array([[700, 0, 1300, 600, 0.99]])]]

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scores = get_label_quality_scores(labels, predictions)

    np.testing.assert_allclose(scores, [1.0])
