
# coding: utf-8

# In[ ]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[ ]:


# Make sure python version is compatible with fasttext
from cleanlab.util import VersionWarning
python_version = VersionWarning(
    warning_str = "fastText supports Python 3 versions (not python 2).",
    list_of_compatible_versions = [3.4, 3.5, 3.6, 3.7],
)


# In[ ]:


# fasttext only exists for these versions that are also compatible with cleanlab
if python_version.is_compatible(): # pragma: no cover
    import time
    import os
    from sklearn.metrics import accuracy_score
    import numpy as np
    from fastText import train_supervised


# In[ ]:


LABEL = '__label__'


# In[ ]:


def _get_labels_text(
    fn = None,
    indices = None, 
    label = LABEL,
):
    '''Helper function for FastTextClassifier'''

    # Prepare data as a list of strings
    with open(fn, 'rU') as f:
        data = [z.strip() for z in f.readlines()]
    # Split text data into a list of examples
    data = [label+z.strip() for z in ''.join(data).split(label) if len(z) > 0]
    if indices is not None and len(indices) < len(data):
        # Only fetch indices provided by indices.
        data = [data[i] for i in indices]
    # Seperate labels and text
    labels, text = [list(t) for t in zip(*(z.split(" ", 1) for z in data))]
    return (labels, text)


# In[ ]:


from sklearn.base import BaseEstimator
class FastTextClassifier(BaseEstimator): # Inherits sklearn base classifier
     
    def __init__(
        self, 
        train_data_fn,
        test_data_fn,
        tmp_dir = '',
        label = LABEL,
        del_intermediate_data = True,
        kwargs_train_supervised = None,
        p_at_k = 5,
        
    ):
        self.train_data_fn = train_data_fn
        self.test_data_fn = test_data_fn
        self.tmp_dir = tmp_dir
        self.label = label
        self.del_intermediate_data = del_intermediate_data
        self.kwargs_train_supervised = kwargs_train_supervised  
        self.p_at_k = p_at_k
        
        #Find all class labels across the train and test set
        labels = set(_get_labels_text(fn = train_data_fn)[0])
        labels = labels.union(set(_get_labels_text(fn = test_data_fn)[0]))
        # Create maps: label strings <-> integers when label strings are used
        labels = sorted(list(labels))
        self.label2num = dict(zip(labels, range(len(labels))))
        self.num2label = dict((y,x) for x,y in self.label2num.items())
    
    
    def _create_train_data(self, data_indices):
        '''Returns filename of the masked fasttext data file.'''
        
        # If X indexes all training data, no need to rewrite the file.
        if data_indices is None:
            self.masked_data_was_created = False
            return self.train_data_fn
        # Mask training data by data_indices
        else:
            # Read in training data as a list of strings
            with open(self.train_data_fn, 'rU') as f:
                data = f.readlines()
            # Split into each training example
            data = [self.label+z for z in ''.join(data).split(self.label) if len(z) > 0]
            # If X indexes all training data, no need to rewrite the file.
            if len(data_indices) == len(data):
                self.masked_data_was_created = False
                return self.train_data_fn
            # Mask by data_indices
            masked_data = [data[i] for i in data_indices]
            masked_fn = "fastTextClf_" + str(int(time.time())) + ".txt"
            with open(masked_fn, 'w') as f:
                f.writelines(masked_data)
            self.masked_data_was_created = True
                
        return masked_fn
    
    
    def _remove_masked_data(self, fn):
        '''docstring'''
        
        if self.del_intermediate_data and self.masked_data_was_created:
            os.remove(fn)
    
    
    def fit(self, X = None, y = None, sample_weight = None):
        '''docstring'''
        
        train_fn = self._create_train_data(data_indices = X)
        self.clf = train_supervised(train_fn, **self.kwargs_train_supervised)
        self._remove_masked_data(train_fn)
    
    
    def predict_proba(self, X = None, train_data = True, return_labels = False):
        '''docstring'''
        
        fn = self.train_data_fn if train_data else self.test_data_fn
        gold_labels, text = _get_labels_text(fn = fn, indices = X)
        pred = self.clf.predict(text = text, k = len(self.clf.get_labels()))
        # Get p(s = k | x) matrix of shape (N x K) of pred probs for each example & class label
        psx = [[p for _, p in sorted(list(zip(*l)), key=lambda x: x[0])] for l in list(zip(*pred))]
        psx = np.array(psx)
        if return_labels:
            return (psx, np.array([self.label2num[y] for y in gold_labels]))
        else:
            return psx
    
        
    def predict(self, X = None, train_data = True, return_labels = False):
        '''Predict labels of X'''
        
        fn = self.train_data_fn if train_data else self.test_data_fn
        gold_labels, text = _get_labels_text(fn = fn, indices = X)
        pred = np.array([self.label2num[z[0]] for z in self.clf.predict(text)[0]])
        if return_labels:
            return (pred, np.array([self.label2num[y] for y in gold_labels]))
        else:
            return pred
    
    
    def score(self, X = None, y = None, sample_weight = None, k = None):
        '''Compute the average precision @ k (single label) of the 
        labels predicted from X and the true labels given by y.'''
        
        # Set the k for precision@k. For single label: 1 if label is in top k, else 0
        if k is None:
            k = self.p_at_k
            
        test_labels, text = _get_labels_text(fn = self.test_data_fn, indices = X)
        y = test_labels if y is None else [self.num2label[z] for z in y]
        apk = np.mean([y[i] in l for i, l in enumerate(self.clf.predict(text, k = k)[0])])
        
        return apk

