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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class NearDuplicateIssueManager(IssueManager):
    """Manages issues related to near-duplicate examples."""

    description: ClassVar[
        str
    ] = """A (near) duplicate issue refers to two or more examples in
    a dataset that are extremely similar to each other, relative
    to the rest of the dataset.  The examples flagged with this issue
    may be exactly duplicated, or lie atypically close together when
    represented as vectors (i.e. feature embeddings).
    """
    issue_name: ClassVar[str] = "near_duplicate"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["threshold"],
    }

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = None,
        threshold: float = 0.13,
        k: int = 10,
        **_,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.threshold = self._set_threshold(threshold)
        self.k = k
        self.near_duplicate_sets: List[List[int]] = []

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        knn_graph = self._process_knn_graph_from_inputs(kwargs)
        old_knn_metric = self.datalab.get_info("statistics").get("knn_metric")
        metric_changes = self.metric and self.metric != old_knn_metric

        if knn_graph is None or metric_changes:
            if features is None:
                raise ValueError(
                    "If a knn_graph is not provided, features must be provided to fit a new knn."
                )
            if self.metric is None:
                self.metric = "cosine" if features.shape[1] > 3 else "euclidean"
            knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)

            if self.metric and self.metric != knn.metric:
                warnings.warn(
                    f"Metric {self.metric} does not match metric {knn.metric} used to fit knn. "
                    "Most likely an existing NearestNeighbors object was passed in, but a different "
                    "metric was specified."
                )
            self.metric = knn.metric

            try:
                check_is_fitted(knn)
            except NotFittedError:
                knn.fit(features)

            knn_graph = knn.kneighbors_graph(mode="distance")
        N = knn_graph.shape[0]
        nn_distances = knn_graph.data.reshape(N, -1)[:, 0]
        median_nn_distance = max(
            np.median(nn_distances), np.finfo(np.float_).eps
        )  # avoid threshold = 0
        self.near_duplicate_sets = self._neighbors_within_radius(
            knn_graph, self.threshold, median_nn_distance
        )

        # Flag every example in a near-duplicate set as a near-duplicate issue
        all_near_duplicates = np.unique(np.concatenate(self.near_duplicate_sets))
        is_issue_column = np.zeros(N, dtype=bool)
        is_issue_column[all_near_duplicates] = True
        temperature = 1.0 / median_nn_distance
        scores = _compute_scores_with_exp_transform(nn_distances, temperature=temperature)
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=scores.mean())
        self.info = self.collect_info(knn_graph=knn_graph, median_nn_distance=median_nn_distance)

    @staticmethod
    def _neighbors_within_radius(knn_graph: csr_matrix, threshold: float, median: float):
        """Returns a list of lists of indices of near-duplicate examples.

        Each list of indices represents a set of near-duplicate examples.

        If the list is empty for a given example, then that example is not
        a near-duplicate of any other example.
        """

        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)
        # Create a mask for the threshold
        mask = distances < threshold * median

        # Update the indptr to reflect the new number of neighbors
        indptr = np.zeros(knn_graph.indptr.shape, dtype=knn_graph.indptr.dtype)
        indptr[1:] = np.cumsum(mask.sum(axis=1))

        # Filter the knn_graph based on the threshold
        indices = knn_graph.indices[mask.ravel()]
        near_duplicate_sets = [indices[indptr[i] : indptr[i + 1]] for i in range(N)]

        # Second pass over the data is required to ensure each item is included in the near-duplicate sets of its own near-duplicates.
        # This is important because a "near-duplicate" relationship is reciprocal.
        # For example, if item A is a near-duplicate of item B, then item B should also be considered a near-duplicate of item A.
        # NOTE: This approach does not assure that the sets are ordered by increasing distance.
        for i, near_duplicates in enumerate(near_duplicate_sets):
            for j in near_duplicates:
                if i not in near_duplicate_sets[j]:
                    near_duplicate_sets[j] = np.append(near_duplicate_sets[j], i)

        return near_duplicate_sets

    def _process_knn_graph_from_inputs(self, kwargs: Dict[str, Any]) -> Union[csr_matrix, None]:
        """Determine if a knn_graph is provided in the kwargs or if one is already stored in the associated Datalab instance."""
        knn_graph_kwargs: Optional[csr_matrix] = kwargs.get("knn_graph", None)
        knn_graph_stats = self.datalab.get_info("statistics").get("weighted_knn_graph", None)

        knn_graph: Optional[csr_matrix] = None
        if knn_graph_kwargs is not None:
            knn_graph = knn_graph_kwargs
        elif knn_graph_stats is not None:
            knn_graph = knn_graph_stats

        if isinstance(knn_graph, csr_matrix) and kwargs.get("k", 0) > (
            knn_graph.nnz // knn_graph.shape[0]
        ):
            # If the provided knn graph is insufficient, then we need to recompute the knn graph
            # with the provided features
            knn_graph = None
        return knn_graph

    def collect_info(self, knn_graph: csr_matrix, median_nn_distance: float) -> dict:
        issues_dict = {
            "average_near_duplicate_score": self.issues[self.issue_score_key].mean(),
            "near_duplicate_sets": self.near_duplicate_sets,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
        }

        N = knn_graph.shape[0]
        dists = knn_graph.data.reshape(N, -1)[:, 0]
        nn_ids = knn_graph.indices.reshape(N, -1)[:, 0]

        knn_info_dict = {
            "nearest_neighbor": nn_ids.tolist(),
            "distance_to_nearest_neighbor": dists.tolist(),
            "median_distance_to_nearest_neighbor": median_nn_distance,
        }

        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)

        info_dict = {
            **issues_dict,
            **params_dict,
            **knn_info_dict,
            **statistics_dict,
        }
        return info_dict

    def _build_statistics_dictionary(self, knn_graph: csr_matrix) -> Dict[str, Dict[str, Any]]:
        statistics_dict: Dict[str, Dict[str, Any]] = {"statistics": {}}

        # Add the knn graph as a statistic if necessary
        graph_key = "weighted_knn_graph"
        old_knn_graph = self.datalab.get_info("statistics").get(graph_key, None)
        old_graph_exists = old_knn_graph is not None
        prefer_new_graph = (
            not old_graph_exists
            or knn_graph.nnz > old_knn_graph.nnz
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            statistics_dict["statistics"][graph_key] = knn_graph
            if self.metric is not None:
                statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

    def _set_threshold(
        self,
        threshold: float,
    ) -> float:
        """Computes nearest-neighbors thresholding for near-duplicate detection."""
        if threshold < 0:
            warnings.warn(
                f"Computed threshold {threshold} is less than 0. "
                "Setting threshold to 0."
                "This may indicate that either the only a few examples are in the dataset, "
                "or the data is heavily skewed."
            )
            threshold = 0
        return threshold


def _compute_scores_with_exp_transform(nn_distances: np.ndarray, temperature: float) -> np.ndarray:
    r"""Compute near-duplicate scores from nearest neighbor distances.

    This is a non-linear transformation of the nearest neighbor distances that
    maps distances to scores in the range [0, 1].

    Note
    ----

    This transformation is given by the following formula:

    .. math::

        \text{score}(d, t) = 1 - e^{-dt}

    where :math:`d` is the nearest neighbor distance and :math:`t > 0` is a temperature parameter.

    Parameters
    ----------
    nn_distances :
        The nearest neighbor distances for each example.

    Returns
    -------
    scores :
        The near-duplicate scores for each example. The scores are in the range [0, 1].
        A lower score indicates that an example is more likely to be a near-duplicate than
        an example with a higher score.
        A score of 0 indicates that an example has an exact duplicate.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    scores = 1 - np.exp(-temperature * nn_distances)
    return scores
