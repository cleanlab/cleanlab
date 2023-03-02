from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional
import warnings

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cleanlab.experimental.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.datalab import Datalab


# TODO typing and method signatures


def simplified_kolmogorov_smirnov_test(
    neighbor_histogram: npt.NDArray[np.float64],
    non_neighbor_histogram: npt.NDArray[np.float64],
) -> float:  # TODO change name
    """Computes the Kolmogorov-Smirnov statistic between two groups of data.
    The statistic is the largest difference between the empirical cumulative
    distribution functions (ECDFs) of the two groups.

    Parameters
    ----------
    neighbor_histogram :
       Histogram data for the nearest neighbor group.

    non_neighbor_histogram :
        Histogram data for the non-neighbor group.

    Returns
    -------
    statistic :
        The KS statistic between the two ECDFs.

    Note
    ----
    - Both input arrays should have the same length.
    - The input arrays are histograms, which means they contain the count
      or frequency of values in each group. The data in the histograms
      should be normalized so that they sum to one.

    To calculate the KS statistic, the function first calculates the ECDFs
    for both input arrays, which are step functions that show the cumulative
    sum of the data up to each point. The function then calculates the
    largest absolute difference between the two ECDFs.
    """

    neighbor_cdf = np.cumsum(neighbor_histogram)
    non_neighbor_cdf = np.cumsum(non_neighbor_histogram)

    statistic = np.max(np.abs(neighbor_cdf - non_neighbor_cdf))
    return statistic


class NonIIDIssueManager(IssueManager):
    """Manages issues related to non-iid data distributions.

    Parameters
    ----------
    datalab :
        The Datalab instance that this issue manager searches for issues in.

    metric :
        The distance metric used to compute the KNN graph of the examples in the dataset.
        If set to `None`, the metric will be automatically selected based on the dimensionality
        of the features used to represent the examples in the dataset.

    k :
        The number of nearest neighbors to consider when computing the KNN graph of the examples.

    num_permutations :
        The number of trials to run when performing permutation testing to determine whether
        the distribution of index-distances between neighbors in the dataset is IID or not.

    """

    description: ClassVar[
        str
    ] = """The NonIIDIssueManager detects whether the given dataset is sampled IID or not.

    Data can be non-IID in many ways and in general it is impossible to detect all cases of non-IID sampling.
    This issue manager investigates whether the ordering of examples in the dataset is dependent on whether
    examples are neighbors in the KNN graph or not. The algorithm uses permutation testing with the 
    Kolmogorov-Smirnov statistic to determine whether the distribution of index-distances between neighbors
    in the dataset is significantly different than that of the non-neighbors in the dataset.

    Detecting non-IID data can very important when collecting datasets or preparing a model for deployment.
    Although shuffling data is generally good practice for removing non-IID issues, knowing that there are
    underlying problems with distribution drift or dependent sampling during data collection is important to
    know in order to understand the real-world environment that your model will be deployed in.

    Types of non-IID problems in datasets can be:
        - Distribution drift or concept drift
        - Video frames in an image dataset
        - Sorting
        - Dependent sampling
    """
    issue_name: ClassVar[str] = "non_iid"
    verbosity_levels = {
        0: {"info": ["p-value"]},
        1: {},
        2: {"issue": ["nearest_neighbor", "distance_to_nearest_neighbor"]},
    }

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[str] = None,
        k: int = 10,
        num_permutations: int = 10,
        **_,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.k = k
        self.num_permutations = num_permutations
        self.knn: Optional[NearestNeighbors] = None
        self.tests = {
            "ks": simplified_kolmogorov_smirnov_test,
        }
        self._histogram1d = None

    def find_issues(self, features: npt.NDArray, **_) -> None:
        if self.knn is None:
            if self.metric is None:
                self.metric = "cosine" if features.shape[1] > 3 else "euclidean"
            self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)

        if self.metric and self.metric != self.knn.metric:
            warnings.warn(
                f"Metric {self.metric} does not match metric {self.knn.metric} used to fit knn. "
                "Most likely an existing NearestNeighbors object was passed in, but a different "
                "metric was specified."
            )
        self.metric = self.knn.metric

        try:
            check_is_fitted(self.knn)
        except:
            self.knn.fit(features)

        self.neighbor_graph = self._get_neighbor_graph(self.knn)

        self.num_neighbors = self.k
        self.num_non_neighbors = min(
            10 * self.num_neighbors, len(self.neighbor_graph) - self.num_neighbors - 1
        )
        self.neighbor_index_distances = self._sample_neighbors(num_samples=self.num_neighbors)
        self.non_neighbor_index_distances = self._sample_non_neighbors(
            num_samples=self.num_non_neighbors
        )
        neighbor_histogram = self._build_histogram(self.neighbor_index_distances.flatten())
        non_neighbor_histogram = self._build_histogram(self.non_neighbor_index_distances.flatten())

        self.statistics = self._get_statistics(neighbor_histogram, non_neighbor_histogram)

        self.p_value = self._permutation_test(num_permutations=self.num_permutations)

        # TODO what about scores?
        scores = self._score_dataset()
        score_median_threshold = np.median(scores) * 0.7
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < score_median_threshold,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(
            score=self.p_value
        )  # TODO is the p-value the right thing to include here?

        self.info = self.collect_info()

    def collect_info(self) -> dict:
        issues_dict = {
            "p-value": self.p_value,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
        }

        weighted_knn_graph = self.knn.kneighbors_graph(mode="distance")  # type: ignore[union-attr]

        knn_info_dict = {
            "weighted_knn_graph": weighted_knn_graph.toarray().tolist(),
        }

        info_dict = {
            **issues_dict,
            **params_dict,  # type: ignore[arg-type]
            **knn_info_dict,
        }
        return info_dict

    def _get_histogram1d(self):
        # TODO: Test correctness of self._histogram1d()(test_array, test_num_bins, test_bin_range)
        if self._histogram1d is None:
            try:
                from fast_histogram import histogram1d as _histogram1d
            except ImportError as e:

                def _histogram1d(array, num_bins, bin_range):
                    return np.histogram(array, num_bins, range=bin_range)[0]

            self._histogram1d = _histogram1d
        return self._histogram1d

    def _permutation_test(self, num_permutations) -> float:
        N = len(self.neighbor_graph)
        tiled = np.tile(np.arange(N), (N, 1))
        index_distances = tiled - tiled.T

        statistics = []
        for _ in range(num_permutations):
            perm = np.random.permutation(N)
            distance = (perm - np.arange(N)).reshape(N, 1)
            new_distances = np.abs(distance - index_distances - distance.transpose())

            neighbor_index_distances = self._sample_neighbors(
                distances=new_distances, num_samples=self.num_neighbors
            ).flatten()
            non_neighbor_index_distances = self._sample_non_neighbors(
                distances=new_distances, num_samples=self.num_non_neighbors
            ).flatten()
            neighbor_histogram = self._build_histogram(neighbor_index_distances)
            non_neighbor_histogram = self._build_histogram(non_neighbor_index_distances)

            stats = self._get_statistics(
                neighbor_histogram,
                non_neighbor_histogram,
            )
            statistics.append(stats)

        ks_stats = np.array([stats["ks"] for stats in statistics])
        ks_stats_kde = gaussian_kde(ks_stats)
        p_value = ks_stats_kde.integrate_box(self.statistics["ks"], 100)

        return p_value

    def _score_dataset(self) -> npt.NDArray[np.float64]:
        graph = self.neighbor_graph
        scores = {}

        num_bins = len(graph) - 1
        bin_range = (1, num_bins)

        neighbor_cdfs = self._compute_row_cdf(self.neighbor_index_distances, num_bins, bin_range)
        non_neighbor_cdfs = self._compute_row_cdf(
            self.non_neighbor_index_distances, num_bins, bin_range
        )

        stats = np.sum(np.abs(neighbor_cdfs - non_neighbor_cdfs), axis=1)

        indices = np.arange(len(graph))
        reverse = len(graph) - indices
        normalizer = np.where(indices > reverse, indices, reverse)

        scores = stats / normalizer
        scores = np.tanh(-1 * scores) + 1
        return scores

    def _compute_row_cdf(self, array, num_bins, bin_range) -> np.ndarray:
        histogram1d = self._get_histogram1d()
        histograms = np.apply_along_axis(lambda x: histogram1d(x, num_bins, bin_range), 1, array)
        histograms = histograms / np.sum(histograms[0])

        cdf = np.apply_along_axis(np.cumsum, 1, histograms)
        return cdf

    def _get_neighbor_graph(self, knn: NearestNeighbors) -> np.ndarray:
        """
        Given a fitted knn object, returns an array in which A[i,j] = n if
        item i and j are nth nearest neighbors. For n > k, A[i,j] = -1. Additionally, A[i,i] = 0
        """

        distances, kneighbors = knn.kneighbors()
        graph = knn.kneighbors_graph(n_neighbors=self.k).toarray()

        kneighbor_graph = np.ones(graph.shape) * -1
        for i, nbrs in enumerate(kneighbors):
            kneighbor_graph[i, nbrs] = 1 + np.arange(len(nbrs))
            kneighbor_graph[i, i] = 0
        return kneighbor_graph

    def _get_statistics(
        self,
        neighbor_index_distances,
        non_neighbor_index_distances,
    ) -> dict[str, float]:
        statistics = {}
        for key, test in self.tests.items():
            statistic = test(
                neighbor_index_distances,
                non_neighbor_index_distances,
            )
            statistics[key] = statistic
        return statistics

    def _sample_distances(self, sample_neighbors, distances=None, num_samples=1) -> np.ndarray:
        graph = self.neighbor_graph
        N = len(graph)
        all_idx = np.arange(N)
        all_idx = np.tile(all_idx, (N, 1))
        if sample_neighbors:
            indices = all_idx[graph > 0].reshape(N, -1)
        else:
            indices = all_idx[graph < 0].reshape(N, -1)
        generator = np.random.default_rng()
        choices = generator.choice(indices, axis=1, size=num_samples, replace=False)
        if distances is None:
            sample_distances = np.abs(np.arange(N) - choices.transpose()).transpose()
        else:
            sample_distances = distances[np.arange(N), choices.transpose()].transpose()
        return sample_distances

    def _sample_neighbors(self, distances=None, num_samples=1) -> np.ndarray:
        return self._sample_distances(
            sample_neighbors=True, distances=distances, num_samples=num_samples
        )

    def _sample_non_neighbors(self, distances=None, num_samples=1) -> np.ndarray:
        return self._sample_distances(
            sample_neighbors=False, distances=distances, num_samples=num_samples
        )

    def _build_histogram(self, index_array) -> np.ndarray:
        histogram1d = self._get_histogram1d()
        num_bins = len(self.neighbor_graph) - 1
        bin_range = (1, num_bins)
        histogram = histogram1d(index_array, num_bins, bin_range)
        histogram = histogram / len(index_array)
        return histogram
