import pandas as pd
import numpy as np
from cleanlab.rank import get_label_quality_scores as main_get_label_quality_scores


def get_label_quality_scores(labels: list, pred_probs:list, *, 
    tokens: list = None, 
    token_scoring_method: str = "self_confidence",                
    sentence_scoring_method: str = "min",
    param: float = 0.04,
    return_scores_per_token: bool = True,
):
    """Returns label quality scores for each sentence

    This is a function to compute label-quality scores for token classification datasets, where lower scores
    indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).
    
    If `return_scores_per_token` is set to True, also return label score per token

    Parameters
    ----------
    labels: list
        noisy token labels in nested list format, such that `labels[i]` is a list of token labels of the i'th
        sentence. For datasets with `K` classes, each label must be in 0, 1, ..., K-1. All classes must be present. 

    pred_probs: list
        list of np.arrays, such that `pred_probs[i]` is the model-predicted probabilities for the tokens in
        the i'th sentence, and has shape `(N, K)`. Each row of the matrix corresponds to a token `t` and contains
        the model-predicted probabilities that `t` belongs to each possible class, for each of the K classes. The
        columns must be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    sentence_scoring_method: {"min", "softmin"}, default="min"
        sentence scoring method to aggregate token scores.

        - `min`: sentence score = minimum token label score of the sentence
        - `softmin`: sentence score = <s, softmax(1-s, t)>, where s denotes the token label scores of the sentence,
        and <a, b> == np.dot(a, b). The parameter `t` controls parameter of softmax, such that when t -> 0, the
        method approaches to `min`. This method is the "softer" version of `min`, which adds some minor weights to
        other scores.

    token_scoring_method: {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
        label quality scoring method. See `cleanlab.rank.get_label_quality_scores` for more info.

    param: float, default=0.04
        temperature of softmax. If sentence_scoring_method == "min", `param` is ignored.

    tokens: list, optinal, default=None
        tokens in nested list format, such that `tokens[i]` is a list of tokens for the i'th sentence. See return value
        `token_info` for more info.

    return_scores_per_token: bool, default=True
        If set to True, returns additional token information. See return value `token_info` for more info.

    Returns
    ----------
    sentence_scores: np.array
        A vector of sentence scores between 0 and 1, where lower scores indicate sentence is more likely to contain at
        least one label issue.

    token_info: list
        Returns only if `return_scores_per_token=True`. A list of pandas.Series, such that token_info[i] contains the 
        token scores for the i'th sentence. If tokens are provided, the series is indexed by the tokens.

    ----------
    """
    methods = ["min", "softmin"]
    assert sentence_scoring_method in methods, "Select from the following methods:\n%s" % "\n".join(methods)

    labels_flatten = np.array([l for label in labels for l in label])
    pred_probs_flatten = np.array([p for pred_prob in pred_probs for p in pred_prob])
    n, m = pred_probs_flatten.shape

    # TODO: maybe move this to a helper in the module? may be helpful for users
    sentence_length = [len(label) for label in labels]

    def nested_list(x, length):
        i = iter(x)
        return [[next(i) for _ in range(length)] for length in sentence_length]

    token_scores = main_get_label_quality_scores(
        labels=labels_flatten, pred_probs=pred_probs_flatten, method=token_scoring_method 
    )
    scores_nl = nested_list(token_scores, sentence_length)

    if sentence_scoring_method == "min":
        fun = lambda scores: np.min(scores)

    elif sentence_scoring_method == "softmin":
        softmax = lambda scores: np.exp(scores / param) / np.sum(np.exp(scores / param))
        fun = lambda scores: np.dot(scores, softmax(1 - np.array(scores)))

    sentence_scores = list(map(fun, scores_nl))
    sentence_scores = np.array(sentence_scores)

    if not return_scores_per_token:
        return sentence_scores

    if tokens:
        token_info = [pd.Series(scores, index=token) for scores, token in zip(scores_nl, tokens)]
    else:
        token_info = [pd.Series(scores) for scores in scores_nl]
    return sentence_scores, token_info
