import string
import numpy as np
from termcolor import colored


def get_sentence(words):
    sentence = ""
    for word in words:
        if word not in string.punctuation or word in ["-", "("]:
            word = " " + word
        sentence += word
    sentence = sentence.replace(" '", "'").replace("( ", "(").strip()
    return sentence


def filter_sentence(sentences, condition=None, return_mask=True):
    if not condition:
        condition = lambda sentence: len(sentence) > 1 and "#" not in sentence
    mask = list(map(condition, sentences))
    sentences = [sentence for m, sentence in zip(mask, sentences) if m]
    if return_mask:
        return sentences, mask
    else:
        return sentences


def process_token(token, replace=[("#", "")]):
    for old, new in replace:
        token = token.replace(old, new)
    return token


def mapping(entities, maps):
    f = lambda x: maps[x]
    return list(map(f, entities))


def merge_probs(probs, maps):
    old_classes = probs.shape[1]
    probs_merged = np.zeros([len(probs), np.max(maps) + 1])

    for i in range(old_classes):
        if maps[i] >= 0:
            probs_merged[:, maps[i]] += probs[:, i]
    if -1 in maps:
        row_sums = probs_merged.sum(axis=1)
        probs_merged /= row_sums[:, np.newaxis]
    return probs_merged


def color_sentence(sentence, word):
    start_idx = sentence.index(word)
    before, after = sentence[:start_idx], sentence[start_idx + len(word) :]
    return "%s%s%s" % (before, colored(word, "red"), after)
