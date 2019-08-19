#!/usr/bin/env python
# coding: utf-8

# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

# Make sure python version is compatible with fasttext
from cleanlab.util import VersionWarning
python_version = VersionWarning(
    warning_str = "fastText supports Python 3 versions (not python 2).",
    list_of_compatible_versions = [3.5, 3.6, 3.7],
)


def create_cooking_dataset(data_dir = None):
    '''This only needs to be run if you do not have
    fasttext_data/cooking.test.txt
    fasttext_data/cooking.train.txt
    
    Before you can use this method, you need to get the 
    cooking.preprocessed.txt file by running:
    bash get_cooking_stackexchange_data.sh .
    
    This is originally modified from here:
    https://github.com/facebookresearch/fastText/blob/master/tests/fetch_test_data.sh#L111
    '''
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Create data_dir if it does not exist.
    import os, errno, shutil
    try:
        os.makedirs(data_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    # Check if dataset already is available
    train_exists = os.path.isfile(data_dir + 'cooking.train.txt')    
    test_exists = os.path.isfile(data_dir + 'cooking.test.txt')
    if not(train_exists and test_exists):  
        # Dataset is not available, create it.

        import subprocess

        # Start out with cooking.preprocessed.txt by running the code here:
        # https://github.com/facebookresearch/fastText/blob/master/tests/fetch_test_data.sh#L111
        
        # Help travis.CI tests find get_cooking_stackexchange_data.sh
        path = '' if cwd[-5:] == 'tests' else cwd + "/tests/"
        # Fetch stackexchange data
        subprocess.call(
            "bash {}get_cooking_stackexchange_data.sh '{}'".format(path, data_dir), 
            shell = True,
        )

        with open(data_dir + 'cooking/cooking.preprocessed.txt', 'rU') as f:
            cook_data = f.readlines()

        single_label_cook_data = []
        for z in cook_data:
            num_labels = z.count('__label__') 
            z_list = z.split(" ")
            labels = z_list[:num_labels]
            content = " ".join(z_list[num_labels:])
            for l in labels:
                 single_label_cook_data.append(l + " " + content)

        with open(data_dir + 'cooking.train.txt', 'w') as f:
            f.writelines(single_label_cook_data[:-5000])

        with open(data_dir + 'cooking.test.txt', 'w') as f:
            f.writelines(single_label_cook_data[-5000:])

        # Clean-up download files
        shutil.rmtree(data_dir + 'cooking')


if python_version.is_compatible():
    from fastText import train_supervised
    import cleanlab
    from cleanlab.models.fasttext import FastTextClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # Set-up for testing.
    import os
    cwd = os.getcwd()
    DATA_DIR = cwd + '/fasttext_data/'

    # Create train and test datasets for testing.
    create_cooking_dataset(DATA_DIR)

    # Load train text data
    with open(DATA_DIR + 'cooking.train.txt', 'r') as f:
        train_data = [z.strip() for z in f.readlines()]
    y_train, X_train = [list(t) for t in zip(*(z.split(" ", 1) for z in train_data))]

    # Load test text data
    with open(DATA_DIR + 'cooking.test.txt', 'r') as f:
        test_data = [z.strip() for z in f.readlines()]
    y_test, X_test = [list(t) for t in zip(*(z.split(" ", 1) for z in test_data))]

    # Set-up a FastTextClassifier model. Train it for five epochs.
    ftc = FastTextClassifier(
        train_data_fn = DATA_DIR + 'cooking.train.txt', 
        test_data_fn = DATA_DIR + 'cooking.test.txt', 
        kwargs_train_supervised = {
            'epoch': 5,
        },
        del_intermediate_data = True,
    )
    ftc.fit(X = None)


def test_predict_proba_masking():
    
    if python_version.is_compatible():
        psx = ftc.predict_proba(X = [500, 1000, 4999])
        assert(psx.shape[0] == 3)
    assert(True)


def test_predict_masking():
    
    if python_version.is_compatible():
        pred = ftc.predict(X = [500, 1000, 4999])
        assert(pred.shape[0] == 3)
    assert(True)


def test_score_masking():
    
    if python_version.is_compatible():
        score = ftc.score(X = [4, 8,  500, 1000, 4999], k = 5)
        assert(0. <= score <= 1.0)
    assert(True)


def test_apk_strictly_increasing():
    '''apk = average precision @ k. Instead of accuracy,
    we check if the true label is on the top k of the predicted
    labels. Thus, as k is increased, our accuracy should only increase
    or stay the same.'''
    
    if python_version.is_compatible():
        prev_score = 0
        for k in range(1, 10):
            score = ftc.score(X = range(500), k = k)
            assert(score >= prev_score)
            prev_score = score
            print(prev_score)
    assert(True)


def test_predict_and_predict_proba():
    '''Test that we get exactly the same results
    for predicting the class label and the max
    predicted probability for each example regardless
    of if we use the fasttext model or our class to 
    predict the labels and the probabilities.'''
    
    if python_version.is_compatible():
        # Check labels
        us = ftc.predict(train_data = False)
        them = [ftc.label2num[z[0]] for z in ftc.clf.predict(X_test)[0]]
        assert(accuracy_score(us, them) > 0.95)

        # Check probabilities
        us_prob = ftc.predict_proba(train_data = False).max(axis = 1)
        them_prob = ftc.clf.predict(X_test, k = len(us_prob))[1].max(axis = 1)
        assert(np.sum((us_prob - them_prob)**2) < 1e-4)
    assert(True)


def test_correctness():
    '''Test to see if our model produces exactly the 
    same results as the '''
    
    if python_version.is_compatible():
        original = train_supervised(DATA_DIR + 'cooking.train.txt', )

        # Check labels
        us = ftc.predict(train_data = False)
        them = [ftc.label2num[z[0]] for z in original.predict(X_test)[0]]
        # Results should be similar, sans stochasticity (fasttext is not seeded)
        assert(accuracy_score(us, them) > 0.9)

        # Check probabilities
        us_prob = ftc.predict_proba(train_data = False).max(axis = 1)
        them_prob = original.predict(X_test, k = len(us_prob))[1].max(axis = 1)
        # Results should be similar, sans stochasticity (fasttext is not seeded)
        assert(np.mean(abs(us_prob - them_prob)) < 0.1)
    assert(True)


def test_return_labels():
    if python_version.is_compatible():
        # Get predictions, probabilities and labels
        pred, labels1 = ftc.predict(train_data = False, return_labels = True)
        psx, labels2 = ftc.predict_proba(train_data = False, return_labels = True)
        assert(len(pred) == len(labels1))
        assert(all(labels1 == labels2))
    assert(True)


def test_cleanlab_with_fasttext():
    '''Tests FastTextClassifier when used with cleanlab to find a label error.'''
    
    if python_version.is_compatible():
        import cleanlab

        top = 3
        label_counts = list(zip(np.unique(y_train + y_test), cleanlab.util.value_counts(y_train + y_test)))
        # Find which labels occur the most often.
        top_labels = [v for v,c in sorted(label_counts, key=lambda x: x[1])[::-1][:top]]

        # Get indices of data and labels for the top labels
        X_train_idx, y_train_top = [list(w) for w in zip(*[(i, z.split(" ", 1)[0])
            for i, z in enumerate(train_data) if z.split(" ", 1)[0] in top_labels])]
        X_test_idx, y_test_top = [list(w) for w in zip(*[(i, z.split(" ", 1)[0])
            for i, z in enumerate(test_data) if z.split(" ", 1)[0] in top_labels])]

        # Pre-train
        ftc = FastTextClassifier(
            train_data_fn = DATA_DIR + 'cooking.train.txt', 
            test_data_fn = DATA_DIR + 'cooking.test.txt', 
            kwargs_train_supervised = {
                'epoch': 20,
            },
            del_intermediate_data = True,
        )
        ftc.fit(X_train_idx, y_train_top)
        # Set epochs to 1 for getting cross-validated predicted probabilities
        ftc.clf.epoch = 1

        # Dictionary mapping string labels to non-negative integers 0, 1, 2...
        label2num = dict(zip(np.unique(y_train_top), range(top)))
        # Map labels
        s_train = np.array([label2num[z] for z in y_train_top])
        # Compute confident joint and predicted probability matrix for each example
        cj, psx = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(
            X = np.array(X_train_idx), 
            s = s_train, 
            clf = ftc, 
            cv_n_folds=5,
        )
        # Find inidices of errors
        noise_idx = cleanlab.pruning.get_noise_indices(
            s_train, 
            psx, 
            confident_joint=cj, 
        )
        # Extract errors. This works by:
        # (1) masking the training examples we used with the noise indices identified.
        # (2) we find the actual train_data corresponding to those indices.
        errors = np.array(train_data)[np.array(X_train_idx)[noise_idx]]

        # Known error - this should be tagged as substituion, not baking.
        assert('__label__baking what can i use instead of corn syrup ?' in errors)
    assert(True)


def test_create_all_data():
    if python_version.is_compatible():
        fn = ftc._create_train_data(range(len(X_train)))
        with open(fn, 'r') as f:
            assert(len(f.readlines()) == len(X_train))
        os.remove(fn)


def test_score():
    if python_version.is_compatible():
        n = 1000
        y = [ftc.label2num[z] for z in y_test[:n]]
        score = ftc.score(X = range(n), y = y)
        assert(score > 0)
    assert(True)