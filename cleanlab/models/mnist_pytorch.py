# coding: utf-8

# Copyright (c) 2017-2050 Curtis G. Northcutt
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.


# ## A cleanlab compatible PyTorch CNN classifier.
# 
# ## Note to use this model you'll need to have pytorch installed
# See: https://pytorch.org/get-started/locally/


# Python 2 and 3 compatibility
from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement)

# In[ ]:


# Make sure python version is compatible with pyTorch
from cleanlab.util import VersionWarning

python_version = VersionWarning(
    warning_str="pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions=[2.7, 3.5, 3.6, 3.7, 3.8],
)

if python_version.is_compatible():  # pragma: no cover
    from sklearn.base import BaseEstimator
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    from torch.utils.data.sampler import SubsetRandomSampler
    import numpy as np


MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000
SKLEARN_DIGITS_TRAIN_SIZE = 1247
SKLEARN_DIGITS_TEST_SIZE = 550


def get_mnist_dataset(loader):  # pragma: no cover
    """Downloads MNIST as PyTorch dataset.

    Parameters
    ----------
    loader : str (values: 'train' or 'test')."""
    dataset = datasets.MNIST(
        root='../data',
        train=(loader == 'train'),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    return dataset


def get_sklearn_digits_dataset(loader):
    """Downloads Sklearn handwritten digits dataset.
    Uses the last SKLEARN_DIGITS_TEST_SIZE examples as the test
    This is (hard-coded) -- do not change.

    Parameters
    ----------
    loader : str (values: 'train' or 'test')."""
    from torch.utils.data import Dataset
    from sklearn.datasets import load_digits

    class TorchDataset(Dataset):
        """Abstracts a numpy array as a PyTorch dataset."""

        def __init__(self, data, targets, transform=None):
            self.data = torch.from_numpy(data).float()
            self.targets = torch.from_numpy(targets).long()
            self.transform = transform

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.data)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # Get sklearn digits dataset
    X_all, y_all = load_digits(return_X_y=True)
    X_all = X_all.reshape((len(X_all), 8, 8))
    y_train = y_all[:-SKLEARN_DIGITS_TEST_SIZE]
    y_test = y_all[-SKLEARN_DIGITS_TEST_SIZE:]
    X_train = X_all[:-SKLEARN_DIGITS_TEST_SIZE]
    X_test = X_all[-SKLEARN_DIGITS_TEST_SIZE:]
    if loader == 'train':
        return TorchDataset(X_train, y_train, transform=transform)
    elif loader == 'test':
        return TorchDataset(X_test, y_test, transform=transform)
    else:  # prama: no cover
        raise ValueError("loader must be either str 'train' or str 'test'.")


if python_version.is_compatible():
    class Net(nn.Module):
        """Basic Pytorch CNN for MNIST-like data."""

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x, T=1.0):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            return x


class CNN(BaseEstimator):  # Inherits sklearn classifier
    """Wraps a PyTorch CNN for the MNIST dataset within an sklearn template
    by defining .fit(), .predict(), and .predict_proba() functions. This
    template enables the PyTorch CNN to flexibly be used within the sklearn
    architecture -- meaning it can be passed into functions like
    cross_val_predict as if it were an sklearn model. The cleanlab library
    requires that all models adhere to this basic sklearn template and thus,
    this class allows a PyTorch CNN to be used in for learning with noisy
    labels among other things."""

    def __init__(
            self,
            batch_size=64,
            epochs=6,
            log_interval=50,  # Set to None to not print
            lr=0.01,
            momentum=0.5,
            no_cuda=False,
            seed=1,
            test_batch_size=None,
            dataset='mnist',  # you can also choose 'sklearn-digits'
            # Set to 'test' to force fit() and predict_proba() on test_set
            # Be careful setting this, it will override every other loader
            # If you set this to 'test', but call .predict(loader = 'train')
            # then .predict() will still predict on test!
            loader=None,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        if self.cuda:  # pragma: no cover
            torch.cuda.manual_seed(self.seed)

        # Instantiate PyTorch model
        self.model = Net()
        if self.cuda:  # pragma: no cover
            self.model.cuda()

        self.loader_kwargs = {'num_workers': 1,
                              'pin_memory': True} if self.cuda else {}
        self.loader = loader
        if dataset == 'mnist':
            # pragma: no cover
            self.get_dataset = get_mnist_dataset
            self.train_size = MNIST_TRAIN_SIZE
            self.test_size = MNIST_TEST_SIZE
        elif dataset == 'sklearn-digits':
            self.get_dataset = get_sklearn_digits_dataset
            self.train_size = SKLEARN_DIGITS_TRAIN_SIZE
            self.test_size = SKLEARN_DIGITS_TEST_SIZE
        else:  # pragma: no cover
            raise ValueError("dataset must be 'mnist' or 'sklearn-digits'.")
        if test_batch_size is None:
            self.test_batch_size = self.test_size


    def fit(self, train_idx, train_labels=None, sample_weight=None,
            loader='train'):
        """This function adheres to sklearn's "fit(X, y)" format for
        compatibility with scikit-learn. ** All inputs should be numpy
        arrays, not pyTorch Tensors train_idx is not X, but instead a list of
        indices for X (and y if train_labels is None). This function is a
        member of the cnn class which will handle creation of X, y from the
        train_idx via the train_loader. """
        if self.loader is not None:
            loader = self.loader
        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError(
                "Check that train_idx and train_labels are the same length.")

        if sample_weight is not None:  # pragma: no cover
            if len(sample_weight) != len(train_labels):
                raise ValueError("Check that train_labels and sample_weight "
                                 "are the same length.")
            class_weight = sample_weight[
                np.unique(train_labels, return_index=True)[1]]
            class_weight = torch.from_numpy(class_weight).float()
            if self.cuda:
                class_weight = class_weight.cuda()
        else:
            class_weight = None

        train_dataset = self.get_dataset(loader)

        # Use provided labels if not None o.w. use MNIST dataset training labels
        if train_labels is not None:
            # Create sparse tensor of train_labels with (-1)s for labels not
            # in train_idx. We avoid train_data[idx] because train_data may
            # very large, i.e. ImageNet
            sparse_labels = np.zeros(
                self.train_size if loader == 'train' else self.test_size,
                dtype=int) - 1
            sparse_labels[train_idx] = train_labels
            train_dataset.targets = sparse_labels

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            # sampler=SubsetRandomSampler(train_idx if train_idx is not None
            # else range(self.train_size)),
            sampler=SubsetRandomSampler(train_idx),
            batch_size=self.batch_size,
            **self.loader_kwargs
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum)

        # Train for self.epochs epochs
        for epoch in range(1, self.epochs + 1):

            # Enable dropout and batch norm layers
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:  # pragma: no cover
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target).long()
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target, class_weight)
                loss.backward()
                optimizer.step()
                if self.log_interval is not None and \
                        batch_idx % self.log_interval == 0:
                    print(
                        'TrainEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_idx),
                            100. * batch_idx / len(train_loader),
                            loss.item()),
                    )

    def predict(self, idx=None, loader=None):
        """Get predicted labels from trained model."""
        # get the index of the max probability
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)

    def predict_proba(self, idx=None, loader=None):
        if self.loader is not None:
            loader = self.loader
        if loader is None:
            is_test_idx = idx is not None and len(
                idx) == self.test_size and np.all(
                np.array(idx) == np.arange(self.test_size))
            loader = 'test' if is_test_idx else 'train'
        dataset = self.get_dataset(loader)
        # Filter by idx
        if idx is not None:
            if (loader == 'train' and len(idx) != self.train_size) or (
                    loader == 'test' and len(idx) != self.test_size):
                dataset.data = dataset.data[idx]
                dataset.targets = dataset.targets[idx]

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if loader == 'train' else self.test_batch_size,
            **self.loader_kwargs
        )

        # sets model.train(False) inactivating dropout and batch-norm layers
        self.model.eval()

        # Run forward pass on model to compute outputs
        outputs = []
        for data, _ in loader:
            if self.cuda:  # pragma: no cover
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = self.model(data)
            outputs.append(output)

        # Outputs are log_softmax (log probabilities)
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        out = outputs.cpu().numpy() if self.cuda else outputs.numpy()
        pred = np.exp(out)
        return pred
