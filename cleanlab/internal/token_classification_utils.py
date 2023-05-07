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
Helper methods used internally in cleanlab.token_classification
"""
from __future__ import annotations

import re
import string
import numpy as np
from termcolor import colored
from typing import List, Optional, Callable, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    T = TypeVar("T", bound=npt.NBitBase)


def get_sentence(words: List[str]) -> str:
    """
    Get sentence formed by a list of words with minor processing for readability

    Parameters
    ----------
    words:
        list of word-level tokens

    Returns
    ----------
    sentence:
        sentence formed by list of word-level tokens

    Examples
    --------
    >>> from cleanlab.internal.token_classification_utils import get_sentence
    >>> words = ["This", "is", "a", "sentence", "."]
    >>> get_sentence(words)
    'This is a sentence.'
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
    sentences:
        list of sentences

    condition:
        sentence filtering condition

    Returns
    ---------
    sentences:
        list of sentences filtered

    mask:
        boolean mask such that `mask[i] == True` if the i'th sentence is included in the
        filtered sentence, otherwise `mask[i] == False`

    Examples
    --------
    >>> from cleanlab.internal.token_classification_utils import filter_sentence
    >>> sentences = ["Short sentence.", "This is a longer sentence."]
    >>> condition = lambda x: len(x.split()) > 2
    >>> long_sentences, _ = filter_sentence(sentences, condition)
    >>> long_sentences
    ['This is a longer sentence.']
    >>> document = ["# Headline", "Sentence 1.", "&", "Sentence 2."]
    >>> sentences, mask = filter_sentence(document)
    >>> sentences, mask
    (['Sentence 1.', 'Sentence 2.'], [False, True, False, True])
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
    token:
        token which potentially contains special characters

    replace:
        list of tuples `(s1, s2)`, where all occurances of s1 are replaced by s2

    Returns
    ---------
    processed_token:
        processed token whose special character has been replaced

    Note
    ----
        Only applies to characters in the original input token.

    Examples
    --------
    >>> from cleanlab.internal.token_classification_utils import process_token
    >>> token = "#Comment"
    >>> process_token("#Comment")
    'Comment'

    Specify custom replacement rules

    >>> replace = [("C", "a"), ("a", "C")]
    >>> process_token("Cleanlab", replace)
    'aleCnlCb'
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
    entities:
        a list of given entities

    maps:
        a list of mapped entities, such that the i'th indexed token should be mapped to `maps[i]`

    Returns
    ---------
    mapped_entities:
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


def merge_probs(
    probs: npt.NDArray["np.floating[T]"], maps: List[int]
) -> npt.NDArray["np.floating[T]"]:
    """
    Merges model-predictive probabilities with desired mapping

    Parameters
    ----------
    probs:
        A 2D np.array of shape `(N, K)`, where N is the number of tokens, and K is the number of classes for the model

    maps:
        a list of mapped index, such that the probability of the token being in the i'th class is mapped to the
        `maps[i]` index. If `maps[i] == -1`, the i'th column of `probs` is ignored. If `np.any(maps == -1)`, the
        returned probability is re-normalized.

    Returns
    ---------
    probs_merged:
        A 2D np.array of shape ``(N, K')``, where `K'` is the number of new classes. Probabilities are merged and
        re-normalized if necessary.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.token_classification_utils import merge_probs
    >>> probs = np.array([
    ...     [0.55, 0.0125, 0.0375, 0.1, 0.3],
    ...     [0.1, 0.8, 0, 0.075, 0.025],
    ... ])
    >>> maps = [0, 1, 1, 2, 2]
    >>> merge_probs(probs, maps)
    array([[0.55, 0.05, 0.4 ],
           [0.1 , 0.8 , 0.1 ]])
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


def color_sentence(sentence: str, word: str) -> str:
    """
    Searches for a given token in the sentence and returns the sentence where the given token is colored red

    Parameters
    ----------
    sentence:
        a sentence where the word is searched

    word:
        keyword to find in `sentence`. Assumes the word exists in the sentence.
    Returns
    ---------
    colored_sentence:
        `sentence` where the every occurrence of the word is colored red, using ``termcolor.colored``

    Examples
    --------
    >>> from cleanlab.internal.token_classification_utils import color_sentence
    >>> sentence = "This is a sentence."
    >>> word = "sentence"
    >>> color_sentence(sentence, word)
    'This is a \x1b[31msentence\x1b[0m.'

    Also works for multiple occurrences of the word

    >>> document = "This is a sentence. This is another sentence."
    >>> word = "sentence"
    >>> color_sentence(document, word)
    'This is a \x1b[31msentence\x1b[0m. This is another \x1b[31msentence\x1b[0m.'
    """
    colored_word = colored(word, "red")
    return _replace_sentence(sentence=sentence, word=word, new_word=colored_word)


def _replace_sentence(sentence: str, word: str, new_word: str) -> str:
    """
    Searches for a given token in the sentence and returns the sentence where the given token has been replaced by
    `new_word`.

    Parameters
    ----------
    sentence:
        a sentence where the word is searched

    word:
        keyword to find in `sentence`. Assumes the word exists in the sentence.

    new_word:
        the word to replace the keyword with

    Returns
    ---------
    new_sentence:
        `sentence` where the every occurrence of the word is replaced by `colored_word`
    """

    new_sentence, number_of_substitions = re.subn(
        r"\b{}\b".format(re.escape(word)), new_word, sentence
    )
    if number_of_substitions == 0:
        # Use basic string manipulation if regex fails
        new_sentence = sentence.replace(word, new_word)
    return new_sentence
