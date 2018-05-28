# confidentlearning
A Python package for Confident Learning with state-of-the-art algorithms for multiclass learning with noisy labels, latent noisy channel estimation, latent prior estimation, detection of label errors in massive datasets, and much more.

## Note to reviewers only
Please view the supplementary materials of our NIPS2018 submission [by clicking here](https://www.dropbox.com/s/n8hydz9zj6skqwg/nips2018_confident_learning_supplements.pdf?dl=0). The password is the first word of the title of our submission, all lowercased. A system error prevented inclusion at submit-time. 

## Installation
### Both Python 2.7 and Python 3.5 are supported.

To use the **confidentlearning** package just run:

```
$ pip install git+https://github.com/cgnorthcutt/confidentlearning.git
```

If you have issues, you can also clone the repo and install by typing:

```
$ conda update pip # if you use conda
$ git clone https://github.com/cgnorthcutt/confidentlearning.git
$ cd confidentlearning
$ pip install -e .
```

## Get started with easy, quick examples.

New to **confidentlearing**? Start with:

1. [Visualizing confident learning](examples/visualizing_confident_learning.ipynb)
2. [A simple example of learning with noisy labels on the multiclass Iris dataset](examples/iris_simple_example.ipynb). 

These examples show how easy it is to characterize label noise in datasets, learn with noisy labels, identify label errors, estimate latent priors and noisy channels, and more.


## The first practical, general algorithm for multiclass learning with noisy labels
### **confidentlearning** algorithms are fast, robust, and work with any probablistic classifier - see [examples](examples)
Multiclass learning with noisy labels in just three lines of code:

```python
from confidentlearning.classification import RankPruning
from sklearn.linear_model import LogisticRegression as logreg

rp = RankPruning(clf=logreg()) # Pass in any classifier. Yup, neural networks work, too.
rp.fit(X_train, y_may_have_label_errors)
pred = rp.predict(X_test) # Estimates the predictions you would have gotten had you trained without label errors.
``` 
