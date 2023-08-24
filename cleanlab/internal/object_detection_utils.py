# Copyright (C) 2017-2023  Cleanlab Inc.
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

"""
Helper functions used internally for object detection tasks.
"""
from multiprocessing import Pool
from typing import List, Optional, Dict, Any
from collections import defaultdict
import numpy as np

from cleanlab.object_detection.rank import _get_overlap_matrix


def bbox_xyxy_to_xywh(bbox: List[float]) -> Optional[List[float]]:
    """Converts bounding box coodrinate types from x1y1,x2y2 to x,y,w,h"""
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    else:
        print("Wrong bbox shape", len(bbox))
        return None


def softmax(x: np.ndarray, temperature: float = 0.99, axis: int = 0) -> np.ndarray:
    """Gets softmax of scores."""
    x = x / temperature
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def softmin1d(scores: np.ndarray, temperature: float = 0.99, axis: int = 0) -> float:
    """Returns softmin of passed in scores."""
    scores = np.array(scores)
    softmax_scores = softmax(-1 * scores, temperature, axis)
    return np.dot(softmax_scores, scores)


def assert_valid_aggregation_weights(aggregation_weights: Dict[str, Any]) -> None:
    """assert aggregation weights are in the proper format"""
    weights = np.array(list(aggregation_weights.values()))
    if (not np.isclose(np.sum(weights), 1.0)) or (np.min(weights) < 0.0):
        raise ValueError(
            f"""Aggregation weights should be non-negative and must sum to 1.0
                """
        )


def assert_valid_inputs(
    labels: List[Dict[str, Any]],
    predictions,
    method: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Asserts proper input format."""
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions length needs to match. len(labels) == {len(labels)} while len(predictions) == {len(predictions)}."
        )
    # Typecheck labels and predictions
    if not isinstance(labels[0], dict):
        raise ValueError(
            f"Labels has to be a list of dicts. Instead it is list of {type(labels[0])}."
        )
    # check last column of predictions is probabilities ( < 1.)?
    if not isinstance(predictions[0], (list, np.ndarray)):
        raise ValueError(
            f"Prediction has to be a list or np.ndarray. Instead it is type {type(predictions[0])}."
        )
    if not predictions[0][0].shape[1] == 5:
        raise ValueError(
            f"Prediction values have to be of format [x1,y1,x2,y2,pred_prob]. Please refer to the documentation for predicted probabilities under object_detection.rank.get_label_quality_scores for details"
        )

    valid_methods = ["objectlab"]
    if method is not None and method not in valid_methods:
        raise ValueError(
            f"""
            {method} is not a valid object detection scoring method!
            Please choose a valid scoring_method: {valid_methods}
            """
        )

    if threshold is not None and threshold > 1.0:
        raise ValueError(
            f"""
            Threshold is a cutoff of predicted probabilities and therefore should be <= 1.
            """
        )


def calculate_ap_per_class(
    predictions,
    labels,
    iou_threshold=0.5,
    num_procs=4,
):
    num_imgs = len(predictions)
    num_scales = 1
    num_classes = len(predictions[0])
    if num_imgs > 1:
        num_procs = min(num_procs, num_imgs)
        pool = Pool(num_procs)
    ap_per_class_list = []
    for class_num in range(num_classes):
        cls_dets, cls_gts = filter_by_class(predictions, labels, class_num)
        if num_imgs > 1:
            args = []
            tpfp = pool.starmap(
                _get_tp_fp,
                zip(cls_dets, cls_gts, [iou_threshold for _ in range(num_imgs)], *args),
            )
        else:
            tpfp = [
                _get_tp_fp(
                    cls_dets[0],
                    cls_gts[0],
                    iou_threshold,
                )
            ]
        tp, fp = tuple(zip(*tpfp))
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            num_gts[0] += bbox.shape[0]
        cls_dets = np.vstack(cls_dets)
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        recalls = recalls[0, :]
        precisions = precisions[0, :]
        ap = calculate_average_precision(recalls, precisions)
        ap_per_class_list.append(ap)
    if num_imgs > 1:
        pool.close()
    return ap_per_class_list


def filter_by_class(predictions, labels, class_num):
    pred_bboxes = [img_res[class_num] for img_res in predictions]
    lab_bboxes = []
    for label in labels:
        gt_inds = label["labels"] == class_num
        lab_bboxes.append(label["bboxes"][gt_inds, :])
    return pred_bboxes, lab_bboxes


def _get_tp_fp(pred_bboxes, lab_bboxes, iou_threshold=0.5):
    num_dets = pred_bboxes.shape[0]
    num_gts = lab_bboxes.shape[0]
    num_scales = 1
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if lab_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp
    ious = _get_overlap_matrix(pred_bboxes, lab_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-pred_bboxes[:, -1])
    k = 0
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        if ious_max[i] >= iou_threshold:
            matched_gt = ious_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[k, i] = 1
            else:
                fp[k, i] = 1
        else:
            fp[k, i] = 1
    return tp, fp


def calculate_average_precision(recall_list, precision_list):
    recall_list = recall_list[np.newaxis, :]
    precision_list = precision_list[np.newaxis, :]
    num_scales = recall_list.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    zeros = np.zeros((num_scales, 1), dtype=recall_list.dtype)
    ones = np.ones((num_scales, 1), dtype=recall_list.dtype)
    mrec = np.hstack((zeros, recall_list, ones))
    mpre = np.hstack((zeros, precision_list, zeros))
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
    for i in range(num_scales):
        ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
        ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    return ap


def get_per_class_ap(predictions, labels):
    iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    dres = defaultdict(list)
    for thr in iou_thrs:
        b = calculate_ap_per_class(predictions, labels, iou_threshold=thr)
        for j in range(0, len(b)):
            dres[j].append(b[j])
    dm = {}
    for i in dres:
        dm[i] = np.mean(dres[i])
    return dm
