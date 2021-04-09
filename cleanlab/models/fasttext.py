# coding: utf-8

# Python 2 and 3 compatibility
from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement,
)

# Make sure python version is compatible with fasttext
from cleanlab.util import VersionWarning

# fasttext only exists for these versions that are also compatible with cleanlab
# if python_version.is_compatible():  # pragma: no cover
import time
import os
import copy
from sklearn.metrics import accuracy_score
import numpy as np
# You need to install fasttext using pip for this library to work
from fasttext import train_supervised, load_model


python_version = VersionWarning(
    warning_str="fastText supports Python 3 versions (not python 2).",
    list_of_compatible_versions=[3.4, 3.5, 3.6, 3.7, 3.8],
)


LABEL = '__label__'
NEWLINE = ' __newline__ '


def data_loader(
        fn=None,
        indices=None,
        label=LABEL,
        batch_size=1000,
):
    """Returns a generator, yielding two lists containing
    [labels], [text]. Items are always returned in the
    order in the file, regardless if indices are provided."""

    def _split_labels_and_text(batch):
        l, t = [list(t) for t in zip(*(z.split(" ", 1) for z in batch))]
        return l, t

    # Prepare a stack of indices
    if indices is not None:
        stack_indices = sorted(indices, reverse=True)
        stack_idx = stack_indices.pop()

    with open(fn, 'r') as f:
        len_label = len(label)
        idx = 0
        batch_counter = 0
        prev = f.readline()
        batch = []
        while True:
            try:
                line = f.readline()
                line = line
                if line[:len_label] == label or line == '':
                    if indices is None or stack_idx == idx:
                        # Write out prev line and reset prev
                        batch.append(prev.strip().replace('\n', NEWLINE))
                        batch_counter += 1

                        if indices is not None:
                            if len(stack_indices):
                                stack_idx = stack_indices.pop()
                            else:  # No more data in indices, quit loading data.
                                yield _split_labels_and_text(batch)
                                break
                    prev = ''
                    idx += 1
                    if batch_counter == batch_size:
                        yield _split_labels_and_text(batch)
                        # Reset batch
                        batch_counter = 0
                        batch = []
                prev += line
                if line == '':
                    if len(batch) > 0:
                        yield _split_labels_and_text(batch)
                    break
            except EOFError:
                if indices is None or stack_idx == idx:
                    # Write out prev line and reset prev
                    batch.append(prev.strip().replace('\n', NEWLINE))
                    batch_counter += 1
                    yield _split_labels_and_text(batch)
                break


from sklearn.base import BaseEstimator


class FastTextClassifier(BaseEstimator):  # Inherits sklearn base classifier

    def __init__(
            self,
            train_data_fn,
            test_data_fn=None,
            labels=None,
            tmp_dir='',
            label=LABEL,
            del_intermediate_data=True,
            kwargs_train_supervised={},
            p_at_k=1,
            batch_size=1000,

    ):
        self.train_data_fn = train_data_fn
        self.test_data_fn = test_data_fn
        self.tmp_dir = tmp_dir
        self.label = label
        self.del_intermediate_data = del_intermediate_data
        self.kwargs_train_supervised = kwargs_train_supervised
        self.p_at_k = p_at_k
        self.batch_size = batch_size
        self.clf = None

        if labels is None:
            # Find all class labels across the train and test set (if provided)
            unique_labels = set([])
            for labels, _ in data_loader(fn=train_data_fn,
                                         batch_size=batch_size):
                unique_labels = unique_labels.union(set(labels))
            if test_data_fn is not None:
                for labels, _ in data_loader(fn=test_data_fn,
                                             batch_size=batch_size):
                    unique_labels = unique_labels.union(set(labels))
        else:
            # Prepend labels with self.label token (e.g. '__label__'). 
            unique_labels = [label + str(l) for l in labels]
        # Create maps: label strings <-> integers when label strings are used
        unique_labels = sorted(list(unique_labels))
        self.label2num = dict(zip(unique_labels, range(len(unique_labels))))
        self.num2label = dict((y, x) for x, y in self.label2num.items())

    def _create_train_data(self, data_indices):
        '''Returns filename of the masked fasttext data file.
        Items are written in the order they are in the file,
        regardless if indices are provided.'''

        # If X indexes all training data, no need to rewrite the file.
        if data_indices is None:
            self.masked_data_was_created = False
            return self.train_data_fn
        # Mask training data by data_indices
        else:
            len_label = len(LABEL)
            data_indices = sorted(data_indices, reverse=True)
            masked_fn = "fastTextClf_" + str(int(time.time())) + ".txt"
            open(masked_fn, 'w').close()
            # Read in training data one line at a time
            with open(self.train_data_fn, 'rU') as rf:
                idx = 0
                data_idx = data_indices.pop()
                for line in rf:
                    # Mask by data_indices
                    if idx == data_idx:
                        with open(masked_fn, 'a') as wf:
                            wf.write(line.strip().replace('\n', NEWLINE) + "\n")
                        if line[:len_label] == LABEL:
                            if len(data_indices):
                                data_idx = data_indices.pop()
                            else:
                                break
                    # Increment data index if starts with __label__
                    # This enables support for text data containing '\n'.
                    if line[:len_label] == LABEL:
                        idx += 1
            self.masked_data_was_created = True

        return masked_fn

    def _remove_masked_data(self, fn):
        '''Deletes intermediate data files.'''

        if self.del_intermediate_data and self.masked_data_was_created:
            os.remove(fn)

    def __deepcopy__(self, memo):
        if self.clf is None:
            self_clf_copy = None
        else:
            fn = 'tmp_{}.fasttext.model'.format(int(time.time()))
            self.clf.save_model(fn)
            self_clf_copy = load_model(fn)
            os.remove(fn)
        # Store self.clf
        params = self.__dict__
        clf = params.pop('clf')
        # Copy params without self.clf (it can't be copied)
        params_copy = copy.deepcopy(params)
        # Add clf back to self.clf
        self.clf = clf
        # Create copy to return
        clf_copy = FastTextClassifier(self.train_data_fn)
        params_copy['clf'] = self_clf_copy
        clf_copy.__dict__ = params_copy
        return clf_copy

    def fit(self, X=None, y=None, sample_weight=None):
        '''Trains the fast text classifier.
        Typical usage requires NO parameters,
        just clf.fit()  # No params.
        
        Parameters
        ----------
        X : iterable, e.g. list, numpy array (default None)
          The list of indices of the data to use.
          When in doubt, set as None. None defaults to range(len(data)).
        y : None
          Leave this as None. Its a filler to suit sklearns reqs.
        sample_weight : None
          Leave this as None. Its a filler to suit sklearns reqs.'''

        train_fn = self._create_train_data(data_indices=X)
        self.clf = train_supervised(train_fn, **self.kwargs_train_supervised)
        self._remove_masked_data(train_fn)

    def predict_proba(self, X=None, train_data=True, return_labels=False):
        '''Produces a probability matrix with examples on rows and 
        classes on columns, where each row sums to 1 and captures the
        probability of the example belonging to each class.'''

        fn = self.train_data_fn if train_data else self.test_data_fn
        psx_list = []
        if return_labels:
            labels_list = []
        for labels, text in data_loader(fn=fn, indices=X,
                                        batch_size=self.batch_size):
            pred = self.clf.predict(text=text, k=len(self.clf.get_labels()))
            # Get p(s = k | x) matrix of shape (N x K) of pred probs for each x
            psx = [[p for _, p in sorted(list(zip(*l)), key=lambda x: x[0])] for
                   l in list(zip(*pred))]
            psx_list.append(np.array(psx))
            if return_labels:
                labels_list.append(labels)
        psx = np.concatenate(psx_list, axis=0)
        if return_labels:
            gold_labels = [self.label2num[z] for l in labels_list for z in l]
            return (psx, np.array(gold_labels))
        else:
            return psx

    def predict(self, X=None, train_data=True, return_labels=False):
        '''Predict labels of X'''

        fn = self.train_data_fn if train_data else self.test_data_fn
        pred_list = []
        if return_labels:
            labels_list = []
        for labels, text in data_loader(fn=fn, indices=X,
                                        batch_size=self.batch_size):
            pred = [self.label2num[z[0]] for z in self.clf.predict(text)[0]]
            pred_list.append(pred)
            if return_labels:
                labels_list.append(labels)
        pred = np.array([z for l in pred_list for z in l])
        if return_labels:
            gold_labels = [self.label2num[z] for l in labels_list for z in l]
            return (pred, np.array(gold_labels))
        else:
            return pred

    def score(self, X=None, y=None, sample_weight=None, k=None):
        '''Compute the average precision @ k (single label) of the 
        labels predicted from X and the true labels given by y.'''

        # Set the k for precision@k.
        # For single label: 1 if label is in top k, else 0
        if k is None:
            k = self.p_at_k

        fn = self.test_data_fn
        pred_list = []
        if y is None:
            labels_list = []
        for labels, text in data_loader(fn=fn, indices=X,
                                        batch_size=self.batch_size):
            pred = self.clf.predict(text, k=k)[0]
            pred_list.append(pred)
            if y is None:
                labels_list.append(labels)
        pred = np.array([z for l in pred_list for z in l])
        if y is None:
            y = [z for l in labels_list for z in l]
        else:
            y = [self.num2label[z] for z in y]

        apk = np.mean([y[i] in l for i, l in enumerate(pred)])

        return apk
