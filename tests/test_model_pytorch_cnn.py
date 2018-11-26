
# coding: utf-8

# In[1]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[2]:


from cleanlab.models.mnist_pytorch import CNN, MNIST_TEST_SIZE, MNIST_TRAIN_SIZE
import cleanlab
from os.path import expanduser
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import datasets


# In[3]:


# Get home directory to store MNIST dataset
home_dir = expanduser("~")
data_dir = home_dir + "/data/"


# In[4]:


X_train = np.arange(MNIST_TRAIN_SIZE)
X_test = np.arange(MNIST_TEST_SIZE)
# X_train = X_train[X_train % 10 == 0]
y_train = datasets.MNIST(data_dir, train=True, download = True).train_labels.numpy()
y_test = datasets.MNIST(data_dir, train=False, download = True).test_labels.numpy()
py_train = cleanlab.util.value_counts(y_train) / float(len(y_train))
X_test_data = datasets.MNIST(data_dir, train=False, download = True).test_data.numpy()


# In[5]:


def test_mnist_pytorch_cnn(seed = 43):
    
    # pyTorch only exists for these versions that are also compatible with cleanlab
    if sys.version_info[0] in [2.7, 3.5, 3.6]:
        np.random.seed(seed)

        prune_method = 'prune_by_noise_rate'

        # Pre-train
        cnn = CNN(epochs=3, log_interval=1000, loader='test', seed = seed) #pre-train
        cnn.fit(X_test, y_test, loader='test') # pre-train (overfit, not out-of-sample) to entire dataset.

        # Out-of-sample cross-validated holdout predicted probabilities
        np.random.seed(4)
        cnn.epochs = 1 # Single epoch for cross-validation (already pre-trained)
        cj, psx = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(X_test, y_test, cnn, cv_n_folds=2)
        est_py, est_nm, est_inv = cleanlab.latent_estimation.estimate_latent(cj, y_test)
        # algorithmic identification of label errors
        noise_idx = cleanlab.pruning.get_noise_indices(y_test, psx, est_inv, prune_method=prune_method) 

        # Get prediction on test set
        pred = cnn.predict(loader = 'test')

        score = accuracy_score(y_test, pred)
        print(score)

        assert(abs(score - 0.929) < .01)
        
    else:
        assert(True)

