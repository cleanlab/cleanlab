import numpy as np
import pandas as pd
from cleanlab.internal.token_classification_utils import *


def display_issues(
    issues, given_words, *, pred_probs=None, given_labels=None, exclude=[], class_names=None, top=20
):
    """
    Display issues, including sentence with issue token highlighted. Also shows given and predicted label
    if possible.

    Parameters
    ----------
    issues: list
        list of tuples `(i, j)`, which represents the j'th token of the i'th sentence.

    given_words: list
        tokens in a nested-list format, such that `given_words[i]` contains the words of the i'th sentence from
        the original file.

    pred_probs: list, default=None
        list of model-predicted probability, such that `pred_probs[i]` contains the model-predicted probability of
        the tokens in the i'th sentence, and has shape `(N, K)`, where `N` is the number of given tokens of the i'th
        sentence, and `K` is the number of classes predicted by the model. If provided, also displays the predicted
        label of the token.

    given_labels: list, default=None
        list of given labels, such that `given_labels[i]` is a list containing the given labels of the tokens in the
        i'th sentence, and has length equal to the number of given tokens of the i'th sentence. If provided, also
        displays the given label of the token.

    exclude: list, default=[]
        list of given/predicted label swaps to be excluded. For example, if `exclude=[(0, 1), (1, 0)]`, swaps between
        class 0 and 1 are not displayed.

    class_names: list, default=None
        name of classes. If not provided, display the integer index for predicted and given labels.

    top: int, default=20
        maximum number of outputs to be printed.

    """
    if not class_names:
        print(
            "Classes will be printed in terms of their integer index since `class_names` was not provided. "
        )
        print("Specify this argument to see the string names of each class. \n")

    shown = 0
    is_tuple = type(issues[0]) == tuple

    for issue in issues:
        if is_tuple:
            i, j = issue
            sentence = get_sentence(given_words[i])
            word = given_words[i][j]

            if pred_probs:
                prediction = pred_probs[i][j].argmax()
            if given_labels:
                given = given_labels[i][j]
            if pred_probs and given_labels:
                if (given, prediction) in exclude:
                    continue

            if pred_probs and class_names:
                prediction = class_names[prediction]
            if given_labels and class_names:
                given = class_names[given]

            shown += 1
            print("Sentence %d, token %d: \n%s" % (i, j, color_sentence(sentence, word)))
            if given_labels and not pred_probs:
                print("Given label: %s\n" % str(given))
            elif not given_labels and pred_probs:
                print("Predicted label: %s\n" % str(prediction))
            elif given_labels and pred_probs:
                print("Given label: %s, predicted label: %s\n" % (str(given), str(prediction)))
            else:
                print()
        else:
            shown += 1
            sentence = get_sentence(given_words[issue])
            print("Sentence %d: %s\n" % (issue, sentence))
        if shown == top:
            break


def common_label_issues(
    issues,
    given_words,
    *,
    labels=None,
    pred_probs=None,
    class_names=None,
    top=10,
    exclude=[],
    verbose=True
):
    """
    Display the most common tokens that are potentially mislabeled.

    Parameters
    ----------
    issues: list
        list of tuples `(i, j)`, which represents the j'th token of the i'th sentence.

    given_words: list
        tokens in a nested-list format, such that `given_words[i]` contains the words of the i'th sentence from
        the original file.

    labels: list
        list of given labels, such that `labels[i]` is a list containing the given labels of the tokens in the i'th
    sentence, and has length equal to the number of given tokens of the i'th sentence. If provided, also
    displays the given label of the token.

    pred_probs: list
        list of model-predicted probability, such that `pred_probs[i]` contains the model-predicted probability of
    the tokens in the i'th sentence, and has shape `(N, K)`, where `N` is the number of given tokens of the i'th
    sentence, and `K` is the number of classes predicted by the model. If both `labels` and `pred_probs` are
    provided, also evaluate each type of given/predicted label swap.

    class_names: list, default=None
        name of classes. If not provided, display the integer index for predicted and given labels.

    top: int, default=10
        maximum number of outputs to be printed.

    exclude: list, default=[]
        list of given/predicted label swaps to be excluded. For example, if `exclude=[(0, 1), (1, 0)]`, swaps between
        class 0 and 1 are not displayed.

    verbose: bool, default=True
        if set to True, also display each type of given/predicted label swap for each token.

    Returns
    ---------
    df: pandas.DataFrame
        if both `labels` and `pred_probs` are provided, return a data frame with columns ['token', 'given_label',
        'predicted_label', 'num_label_issues'], and each row contains the information for a specific token and
        given/predicted label swap, ordered by the number of label issues in descending order. Otherwise, return
        a data frame with columns ['token', 'num_label_issues'], and each row contains the information for a specific
        token, ordered by the number of label issues in descending order.

    """
    count = {}
    if not labels or not pred_probs:
        for issue in issues:
            i, j = issue
            word = given_words[i][j]
            if word not in count:
                count[word] = 0
            count[word] += 1

        words = [word for word in count.keys()]
        freq = [count[word] for word in words]
        rank = np.argsort(freq)[::-1][:top]

        for r in rank:
            print(
                "Token '%s' is potentially mislabeled %d times throughout the dataset\n"
                % (words[r], freq[r])
            )

        info = [[word, f] for word, f in zip(words, freq)]
        info = sorted(info, key=lambda x: x[1], reverse=True)
        return pd.DataFrame(info, columns=["token", "num_label_issues"])

    if not class_names:
        print(
            "Classes will be printed in terms of their integer index since `class_names` was not provided. "
        )
        print("Specify this argument to see the string names of each class. \n")

    n = pred_probs[0].shape[1]
    for issue in issues:
        i, j = issue
        word = given_words[i][j]
        label = labels[i][j]
        pred = pred_probs[i][j].argmax()
        if word not in count:
            count[word] = np.zeros([n, n], dtype=int)
        if (label, pred) not in exclude:
            count[word][label][pred] += 1
    words = [word for word in count.keys()]
    freq = [np.sum(count[word]) for word in words]
    rank = np.argsort(freq)[::-1][:top]

    for r in rank:
        matrix = count[words[r]]
        most_frequent = np.argsort(count[words[r]].flatten())[::-1]
        print(
            "Token '%s' is potentially mislabeled %d times throughout the dataset"
            % (words[r], freq[r])
        )
        if verbose:
            print(
                "---------------------------------------------------------------------------------------"
            )
            for f in most_frequent:
                i, j = f // n, f % n
                if matrix[i][j] == 0:
                    break
                if class_names:
                    print(
                        "labeled as class `%s` but predicted to actually be class `%s` %d times"
                        % (class_names[i], class_names[j], matrix[i][j])
                    )
                else:
                    print(
                        "labeled as class %d but predicted to actually be class %d %d times"
                        % (i, j, matrix[i][j])
                    )
        print()
    info = []
    for word in words:
        for i in range(n):
            for j in range(n):
                num = count[word][i][j]
                if num > 0:
                    if not class_names:
                        info.append([word, i, j, num])
                    else:
                        info.append([word, class_names[i], class_names[j], num])
    info = sorted(info, key=lambda x: x[3], reverse=True)
    return pd.DataFrame(
        info, columns=["token", "given_label", "predicted_label", "num_label_issues"]
    )


def filter_by_token(token, issues, given_words):
    """
    Searches a specific token within all issue tokens

    Parameters
    ----------
        token: string
            the specific token the user is looking for

        issues: list
            list of tuples `(i, j)`, which represents the j'th token of the i'th sentence.

        given_words: list
            tokens in a nested-list format, such that `given_words[i]` contains the words of the i'th sentence from
        the original file.

    Returns
    ----------
    returned_issues: list
        list of tuples `(i, j)`, which represents the j'th token of the i'th sentence.

    """
    returned_issues = []
    for issue in issues:
        i, j = issue
        if token.lower() == given_words[i][j].lower():
            returned_issues.append(issue)
    return returned_issues
