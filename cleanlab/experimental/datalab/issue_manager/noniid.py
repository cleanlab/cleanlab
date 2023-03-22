from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional
import warnings
import itertools

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cleanlab.experimental.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.datalab import Datalab


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
        num_permutations: int = 25,
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
        self.background_distribution = None
        self.sorted_neighbors = None

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

        self.neighbor_index_choices = self._sample_neighbors(num_samples=self.num_neighbors)

        indices = np.arange(self.N)
        self.neighbor_index_distances = np.abs(indices.reshape(-1, 1) - self.neighbor_index_choices)
        
        self.statistics = self._get_statistics(self.neighbor_index_distances)

        self.p_value = self._permutation_test(num_permutations=self.num_permutations)

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
        )

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
            "weighted_knn_graph": weighted_knn_graph,
        }

        info_dict = {
            **issues_dict,
            **params_dict,  # type: ignore[arg-type]
            **knn_info_dict,
        }
        return info_dict

    def _get_histogram1d(self):
        if self._histogram1d is None:
            try:
                from fast_histogram import histogram1d as _histogram1d
            except ImportError as e:

                def _histogram1d(array, num_bins, bin_range):
                    return np.histogram(array, num_bins, range=bin_range)[0]

            self._histogram1d = _histogram1d
        return self._histogram1d

    def _permutation_test(self, num_permutations) -> float:
        N = self.N

        perms = np.fromiter(itertools.chain.from_iterable(np.random.permutation(N)  for i in range(num_permutations)), dtype=int).reshape(num_permutations, N)


        neighbor_index_choices = self.neighbor_index_choices
        neighbor_index_choices = neighbor_index_choices.reshape(1, *neighbor_index_choices.shape)
        perm_neighbor_choices = perms[:,neighbor_index_choices].reshape(num_permutations, *neighbor_index_choices.shape[1:])
        neighbor_index_distances = np.abs(perms[...,None] - perm_neighbor_choices).reshape(num_permutations, -1)

        statistics = []
        for neighbor_index_dist in neighbor_index_distances:
            stats = self._get_statistics(
                neighbor_index_dist,
            )
            statistics.append(stats)

        ks_stats = np.array([stats["ks"] for stats in statistics])
        ks_stats_kde = gaussian_kde(ks_stats)
        p_value = ks_stats_kde.integrate_box(self.statistics["ks"], 100)

        return p_value


    def _score_dataset(self) -> npt.NDArray[np.float64]:
        """This function computes a variant of the KS statistic for each
        datapoint. Rather than computing the maximum difference
        between the CDF of the neighbor distances (foreground
        distribution) and the CDF of the all index distances
        (background distribution), we compute the absolute difference
        in area-under-the-curve of the two CDFs.

        The foreground distribution is computed by sampling the
        neighbor distances from the KNN graph, but the background
        distribution is computed analytically. The background CDF for
        a datapoint i can be split up into three parts. Let d = min(i,
        N - i - 1).

        1. For 0 < j <= d, the slope of the CDF is 2 / (N - 1) since
        there are two datapoints in the dataset that are distance j
        from datapoint i. We call this threshold the 'double distance
        threshold' 

        2. For d < j <= N - d - 1, the slope of the CDF is
        1 / (N - 1) since there is only one datapoint in the dataset
        that is distance j from datapoint i.

        3. For j > N - d - 1, the slope of the CDF is 0 and is
        constant at 1.0 since there are no datapoints in the dataset
        that are distance j from datapoint i.

        We compute the area differences on each of the k intervals for
        which the foreground CDF is constant which allows for the
        possibility that the background CDF may intersect the
        foreground CDF on this interval. We do not account for these
        cases when computing absolute AUC difference.

        Our algorithm is simple, sort the k sampled neighbor
        distances. Then, for each of the k neighbor distances sampled,
        compute the AUC for each CDF up to that point. Then, subtract
        from each area the previous area in the sorted order to get
        the AUC of the CDF on the interval between those two
        points. Subtract the background interval AUCs from the
        foreground interval AUCs, take the absolute value, and
        sum. The algorithm is vectorized such that this statistic is
        computed for each of the N datapoints simultaneously.

        The statistics are then normalized by their respective maximum
        possible distance (N - d - 1) and then mapped to [0,1] via
        tanh.
        """
        N = self.N

        sorted_neighbors = np.sort(self.neighbor_index_distances, axis=1)

        N = 10
        # find the maximum distance that occurs with double probability
        middle_idx = np.floor((N - 1)/ 2).astype(int)
        double_distances = np.arange(N).reshape(N, 1)
        double_distances[double_distances > middle_idx] -= (N - 1)
        double_distances = np.abs(double_distances)
        print(double_distances)
        foo

        sorted_neighbors = np.hstack([sorted_neighbors, np.ones((N, 1)) * (N-1)]).astype(int)
        
        # the set of distances that are less than the double distance threshold
        set_beginning = sorted_neighbors <= double_distances
        # the set of distances that are greater than the double distance threshold but have nonzero probability
        set_middle = (sorted_neighbors > double_distances) & (sorted_neighbors <= (N - double_distances - 1))
        # the set of distances that occur with 0 probability
        set_end = sorted_neighbors > (N - double_distances - 1)
        
        shifted_neighbors = np.zeros(sorted_neighbors.shape)
        shifted_neighbors[:,1:] = sorted_neighbors[:,:-1]
        diffs = sorted_neighbors - shifted_neighbors # the distances between the sorted indices

        area_beginning = ((double_distances ** 2) / (N - 1))
        length = (N - 2 * double_distances - 1)
        a = 2 * double_distances / (N - 1)
        area_middle = 0.5 * (a + 1) * length


        # compute the area under the CDF for each of the indices in sorted_neighbors
        background_area = np.zeros(diffs.shape)
        background_diffs = np.zeros(diffs.shape)
        background_area[set_beginging] = ((sorted_neighbors ** 2) / (N - 1))[set_beginning]
        background_area[set_middle] = (area_beginning + 0.5 * ((sorted_neighbors + 3 * indices.reshape(N, 1)) * (sorted_neighbors - double_distances) / (N - 1)))[set_middle]
        background_area[set_end] = (area_beginning + area_middle + (sorted_neighbors - (N - double_distances - 1) * 1.0))[set_end]

        # compute the area under the CDF between indices in sorted_neighbors
        shifted_background = np.zeros(background_area.shape)
        shifted_background[:,1:] = background_area[:,:-1]
        background_diffs = background_area - shifted_background

        # compute the foreground CDF and AUC between indices in sorted_neighbors
        foreground_cdf = np.arange(sorted_neighbors.shape[1]) / (sorted_neighbors.shape[1] - 1)
        foreground_diffs = foreground_cdf.reshape(1, -1) * diffs

        # compute the differences between foreground and background area intervals
        area_diffs = np.abs(foreground_diffs - background_diffs)
        stats = np.sum(area_diffs, axis=1)

        # normalize scores by the index and transform to [0, 1]
        indices = np.arange(N)
        reverse = N - indices
        normalizer = np.where(indices > reverse, indices, reverse)

        scores = stats / normalizer
        scores = np.tanh(-1 * scores) + 1
        return scores

    def _compute_row_cdf(self, array, num_bins, bin_range) -> np.ndarray:
        ## TODO use new batch function

        histogram1d = self._get_histogram1d()
        
        histograms = np.apply_along_axis(lambda x: histogram1d(x, num_bins, bin_range), 1, array)
        histograms = histograms / np.sum(histograms[0])
        cdf = np.cumsum(histograms, axis=1)
        return cdf

    def _get_neighbor_graph(self, knn: NearestNeighbors) -> np.ndarray:
        """
        Given a fitted knn object, returns an array in which A[i,j] = n if
        item i and j are nth nearest neighbors. For n > k, A[i,j] = -1. Additionally, A[i,i] = 0
        """

        _, kneighbors = knn.kneighbors()
        graph = knn.kneighbors_graph(n_neighbors=self.k)
        self.N  = graph.shape[0]
        
        # kneighbor_graph = np.ones(graph.shape) * -1
        self.neighbors = kneighbors
        # non_neighbors = np.zeros((self.N, self.N - self.neighbors.shape[1] - 1), dtype=int)
        # indices = np.arange(self.N).reshape(self.N, -1)
        # all_indices = np.tile(indices, (self.N,1))
        # to_delete = np.hstack((self.neighbors, indices))
        
        # non_neighbors = np.delete(all_indices, to_delete).reshape(self.N, -1)
        non_neighbors = None

        
        # print(non_neighbors)
        # for i, nbrs in enumerate(kneighbors):
        #     # kneighbor_graph[i, nbrs] = 1 + np.arange(len(nbrs))
        #     # kneighbor_graph[i, i] = 0
        #     ####
        #     to_delete = np.append(nbrs, i)
        #     non = np.delete(indices, to_delete)
        #     non_neighbors[i] = non
        #self.non_neighbors = np.array(non_neighbors)
        self.non_neighbors = non_neighbors


        return 0 #kneighbor_graph TODO

    def _get_statistics(
        self,
        neighbor_index_distances,
    ) -> dict[str, float]:
        
        if self.sorted_neighbors is None:
            neighbor_index_distances = neighbor_index_distances.flatten()
            self.sorted_neighbors = np.sort(neighbor_index_distances)
            self.sorted_neighbors = np.hstack([self.sorted_neighbors, np.ones((1)) * (self.N-1)]).astype(int)
        sorted_neighbors = self.sorted_neighbors

        if self.background_distribution is None:
            self.background_distribution = (self.N - np.arange(1, self.N)) / (self.N * (self.N - 1) / 2)
            p = 1 / (self.N * (self.N - 1) / 2)
            self.background_levels = np.cumsum(self.background_distribution)

        background_levels = self.background_levels


        # shifted = np.zeros(sorted_neighbors.shape)
        # shifted[1:] = sorted_neighbors[:-1]
        # diffs = sorted_neighbors - shifted


        # background_areas = background_levels[sorted_neighbors - 1] * (sorted_neighbors) / 2
        # shifted_area = np.zeros(background_areas.shape)
        # shifted_area[1:] = background_areas[:-1]
        # background_diffs = np.abs(background_areas - shifted_area)

        foreground_levels = np.arange(sorted_neighbors.shape[0]) / (sorted_neighbors.shape[0] - 1)

        statistic = np.max(np.abs(foreground_levels - background_levels[sorted_neighbors - 1]))
    
        # statistic = np.max(np.abs(foreground_diffs - background_diffs))
        # print(statistic)
        
        
        
        # statistics = {}
        # for key, test in self.tests.items():
        #     statistic = test(
        #         neighbor_index_distances,
        #         non_neighbor_index_distances,
        #     )
        #     statistics[key] = statistic
        statistics = {'ks': statistic}
        return statistics

    def _sample_distances(self, sample_neighbors, distances=None, num_samples=1) -> np.ndarray:
        N = self.N
        # all_idx = np.arange(N)
        # all_idx = np.tile(all_idx, (N, 1))

        # indices = all_idx[graph > 0].reshape(N, -1)
        if sample_neighbors:
            return self.neighbors

        # print(indices.shape)
        # all_idx = np.arange(N)
        # to_delete = np.hstack((indices, np.arange(N).reshape(N, -1)))
        # print(to_delete.shape)
        # print(to_delete)
        # print(all_idx)
        # print('stuff')
        # all_idx[to_delete]
        # print(all_idx[to_delete])
        # indices = np.delete(all_idx, to_delete).reshape(N, -1)
        # print(indices)
        # print(indices.shape)

        # #indices = all_idx[graph < 0].reshape(N, -1)
        indices = self.non_neighbors

        
        generator = np.random.default_rng()
        # indices = np.arange(1, N)
        # def sample(idx):
        #     idx = min(idx, np.abs(self.N - idx - 1))
        #     _max = self.N - idx - 1
        #     p = np.ones(len(indices)) / (N - 1)
        #     p[indices <= idx] = 2 / (N - 1)
        #     p[indices > _max] = 0
        #     return generator.choice(indices, size=num_samples, p=p)

        # choices = np.apply_along_axis(sample, 1, np.arange(N).reshape(N, 1))
        # print(choices)
        # foo
        # print(indices)
        # foo

        # indices = generator.permuted(indices, axis=1)
        # choices = indices[:, :num_samples]

        choices = generator.choice(indices, size=num_samples, axis=1, replace=False, shuffle=False)
        
        return choices
        
        ### inaccurate
        # indices = indices.flatten()
        # num_samples = num_samples * N
        # choices = generator.choice(indices, size=num_samples, replace=False).reshape(N, -1)
        # return choices

    def _sample_neighbors(self, distances=None, num_samples=1) -> np.ndarray:
        return self._sample_distances(
            sample_neighbors=True, distances=distances, num_samples=num_samples
        )

    def _sample_non_neighbors(self, distances=None, num_samples=1) -> np.ndarray:
        return self._sample_distances(
            sample_neighbors=False, distances=distances, num_samples=num_samples,
        )

    def _build_histogram(self, index_array) -> np.ndarray:
        histogram1d = self._get_histogram1d()
        num_bins = self.N - 1
        bin_range = (1, num_bins)
        histogram = histogram1d(index_array, num_bins, bin_range)

        histogram = histogram / len(index_array)
        return histogram

    def _build_histogram_batch(self, index_array) -> np.ndarray:
        histogram1d = self._get_histogram1d()
        num_bins = self.N - 1
        bin_range = (1, num_bins)

        histograms = np.apply_along_axis(lambda x: histogram1d(x, num_bins, bin_range), 1, index_array)

        histograms = histograms / np.sum(histograms[0])
        return histograms

