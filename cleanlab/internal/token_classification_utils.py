import re
import string
import numpy as np
from termcolor import colored
from typing import List, Optional, Callable, Tuple


def get_sentence(words: List[str]) -> str:
    """
    Get sentence formed by a list of words with minor processing for readability

    Parameters
    ----------
    words: List[str]
        list of word-level tokens

    Returns
    ----------
    sentence: string
        sentence formed by list of word-level tokens

    """
    sentence = ""
    for word in words:
        if word not in string.punctuation or word in ["-", "("]:
            word = " " + word
        sentence += word
    sentence = sentence.replace(" '", "'").replace("( ", "(").strip()
    return sentence


def filter_sentence(
    sentences: List[str],
    condition: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[bool]]:
    """
    Filter sentence based on some condition, and returns filter mask

    Parameters
    ----------
        sentences: List[str]
            list of sentences

        condition: Optional[Callable[[str], bool]]
            sentence filtering condition

    Returns
    ---------
        sentences: List[str]
            list of sentences filtered

        mask: List[bool]
            boolean mask such that `mask[i] == True` if the i'th sentence is included in the
            filtered sentence, otherwise `mask[i] == False`

    """
    if not condition:
        condition = lambda sentence: len(sentence) > 1 and "#" not in sentence
    mask = list(map(condition, sentences))
    sentences = [sentence for m, sentence in zip(mask, sentences) if m]
    return sentences, mask


def process_token(token: str, replace: List[Tuple[str, str]] = [("#", "")]) -> str:
    """
    Replaces special characters in the tokens

    Parameters
    ----------
        token: str
            token which potentially contains special characters

        replace: List[Tuple[str, str]]
            list of tuples `(s1, s2)`, where all occurances of s1 are replaced by s2

    Returns
    ---------
        processed_token: str
            processed token whose special character has been replaced

    Note
    ----
        Only applies to characters in the original input token.
    """
    replace_dict = {re.escape(k): v for (k, v) in replace}
    pattern = "|".join(replace_dict.keys())
    compiled_pattern = re.compile(pattern)
    replacement = lambda match: replace_dict[re.escape(match.group(0))]
    processed_token = compiled_pattern.sub(replacement, token)
    return processed_token


def mapping(entities: List[int], maps: List[int]) -> List[int]:
    """
    Map a list of entities to its corresponding entities

    Parameters
    ----------
        entities: List[int]
            a list of given entities

        maps: List[int]
            a list of mapped entities, such that the i'th indexed token should be mapped to `maps[i]`

    Returns
    ---------
        mapped_entities: List[int]
            a list of mapped entities

    Examples
    --------
        >>> unique_identities = [0, 1, 2, 3, 4]  # ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
        >>> maps = [0, 1, 1, 2, 2]  # ["O", "PER", "PER", "LOC", "LOC"]
        >>> mapping(unique_identities, maps)
        [0, 1, 1, 2, 2]  # ["O", "PER", "PER", "LOC", "LOC"]
        >>> mapping([0, 0, 4, 4, 3, 4, 0, 2], maps)
        [0, 0, 2, 2, 2, 2, 0, 1]  # ["O", "O", "LOC", "LOC", "LOC", "LOC", "O", "PER"]
    """
    f = lambda x: maps[x]
    return list(map(f, entities))


def merge_probs(probs: np.ndarray, maps: List[int]) -> np.ndarray:
    """
    Merges model-predictive probabilities with desired mapping

    Parameters
    ----------
        probs:
            np.array of shape `(N, K)`, where N is the number of tokens, and K is the number of classes for the model

        maps: List[int]
            a list of mapped index, such that the probability of the token being in the i'th class is mapped to the
            `maps[i]` index. If `maps[i] == -1`, the i'th column of `probs` is ignored. If `np.any(maps == -1)`, the
            returned probability is re-normalized.

    Returns
    ---------
        probs_merged:
            np.array of shape `(N, K')`, where K' is the number of new classes. Probablities are merged and
            re-normalized if necessary.

    """
    old_classes = probs.shape[1]
    map_size = np.max(maps) + 1
    probs_merged = np.zeros([len(probs), map_size], dtype=probs.dtype.type)

    for i in range(old_classes):
        if maps[i] >= 0:
            probs_merged[:, maps[i]] += probs[:, i]
    if -1 in maps:
        row_sums = probs_merged.sum(axis=1)
        probs_merged /= row_sums[:, np.newaxis]
    return probs_merged


def color_sentence(word: str, tokens: List[str]) -> str:
    """
    Searches for a given token in a list of tokens representing a sentence and returns the sentence where the given token is colored red

    Parameters
    ----------
        word:
            keyword to find in `tokens`. Assumes the word exists in the list of tokens.

        tokens:
            list of tokens that form the sentence.
    Returns
    ---------
        colored_sentence:
            `sentence` where the first occurance of the word is colored red, using `termcolor.colored`

    """
    sentence = get_sentence(tokens)

    idx = -1
    for i, token in enumerate(tokens):
        if token == word:
            idx = i
            break

    if idx == -1:  # not found
        start_idx = sentence.index(word)
        before, after = sentence[:start_idx], sentence[start_idx + len(word) :]

    else:
        before, after = get_sentence(tokens[:idx]), get_sentence(tokens[idx + 1 :])

    return get_sentence([before, colored(word, "red"), after])
