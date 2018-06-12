
# coding: utf-8

# In[2]:

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


# In[ ]:

MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000


# In[ ]:

class Net(nn.Module):
    '''Basic Pytorch CNN'''
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


# In[ ]:

class CNN(object):
    '''Wraps a PyTorch CNN for the MNIST dataset within an sklearn template by defining 
    .fit(), .predict(), and .predict_proba() functions. This template enables the PyTorch
    CNN to flexibly be used within the sklearn architecture -- meaning it can be passed into
    functions like cross_val_predict as if it were an sklearn model. The confidentlearning library
    requires that all models adhere to this basic sklearn template and thus, this class allows
    a PyTorch CNN to be used in for learning with noisy labels among other things.'''
    def __init__(
        self,
        batch_size = 64,
        epochs = 6,
        log_interval = 50, # Set to None to not print
        lr = 0.01,
        momentum = 0.5,
        no_cuda = False,
        seed = 1,
        test_batch_size = MNIST_TEST_SIZE,
        loader = None, # Set to 'test' to force fit() and predict_proba() on test_set
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.test_batch_size = test_batch_size
        
        self.cuda = not self.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        
        # Instantiate PyTorch model
        self.model = Net()
        if self.cuda:
            self.model.cuda()
            
        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.loader = loader

    def fit(self, train_idx, train_labels = None, sample_weight = None, loader = 'train'):
        '''This function adheres to sklearn's "fit(X, y)" format for compatibility with scikit-learn. 
        ** All inputs should be numpy arrays, not pyTorch Tensors
        train_idx is not X, but instead a list of indices for X (and y if train_labels is None).
        This function is a member of the cnn class which will handle creation of X, y from
        the train_idx via the train_loader.'''
        if self.loader is not None:
            loader = self.loader
        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError("Check that train_idx and train_labels are the same length.")
            
        if sample_weight is not None:
            if len(sample_weight) != len(train_labels):
                raise ValueError("Check that train_labels and sample_weight are the same length.")
            class_weight = sample_weight[torch.np.unique(train_labels, return_index=True)[1]]
            class_weight = torch.from_numpy(class_weight).float()
            if self.cuda:
                class_weight = class_weight.cuda()
        else:
            class_weight = None
        
        train_dataset = datasets.MNIST(
            root = '../data', 
            train = (loader=='train'), 
            download = True,
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        
        # Use provided labels if not None, o.w. use MNIST dataset training labels
        if train_labels is not None:
            # Create sparse tensor of train_labels with (-1)s for labels not in train_idx.
            # We avoid train_data[idx] because train_data may very large, i.e. image_net
            sparse_labels = torch.np.zeros(MNIST_TRAIN_SIZE if loader == 'train' else MNIST_TEST_SIZE, dtype=int) - 1
            sparse_labels[train_idx] = train_labels
            train_dataset.train_labels = sparse_labels
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
#             sampler=SubsetRandomSampler(train_idx if train_idx is not None else range(MNIST_TRAIN_SIZE)),
            sampler=SubsetRandomSampler(train_idx), 
            batch_size=self.batch_size, 
            **self.loader_kwargs
        )
            
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            
        # Train for self.epochs epochs
        for epoch in range(1, self.epochs + 1):
            
            # Enable dropout and batch norm layers
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target, class_weight)
                loss.backward()
                optimizer.step()
                if self.log_interval is not None and batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_idx),
                        100. * batch_idx / len(train_loader), loss.data[0]))
    
    def predict(self, idx = None, loader = None):
        # get the index of the max probability
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)
        
        
    def predict_proba(self, idx = None, loader = None):        
        if self.loader is not None:
            loader = self.loader
        if loader is None:
            is_test_idx = (len(idx) == MNIST_TEST_SIZE) and                 (torch.np.array(idx) == torch.np.arange(MNIST_TEST_SIZE)).all()
            loader = 'test' if is_test_idx else 'train'       
        dataset = datasets.MNIST(
            root = '../data', 
            train = (loader=='train'), 
            download = True,
            transform = transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]                                        
            )
        )        
        # Filter by idx
        if idx is not None:
            if loader == 'train' and len(idx) != MNIST_TRAIN_SIZE:
                dataset.train_data = dataset.train_data[idx]
                dataset.train_labels = dataset.train_labels[idx]
            elif loader == 'test' and len(idx) != MNIST_TEST_SIZE:
                dataset.test_data = dataset.test_data[idx]
                dataset.test_labels = dataset.test_labels[idx]            
        
        loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size=self.batch_size if loader=='train' else self.test_batch_size, 
            **self.loader_kwargs
        )
        
        # sets model.train(False) inactivating dropout and batch-norm layers
        self.model.eval()
        
        # Run forward pass on model to compute outputs
        outputs = []
        for data, _ in loader:
            if self.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = self.model(data)
            outputs.append(output)
        
        # Outputs are log_softmax (log probabilities)
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        pred = torch.np.exp(outputs.data.numpy())
        return pred
    
    
    def test(self):
        
        target = datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])).test_labels.numpy()
        
        pred = self.predict(loader = 'test')
        correct = torch.np.count_nonzero(pred == target)
        
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, MNIST_TEST_SIZE,
            100. * correct / MNIST_TEST_SIZE))        

    
    def test_deprecated(self):
        
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=self.test_batch_size, 
            shuffle=True, 
            **self.loader_kwargs
        )
        
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# In[ ]:

if __name__ == '__main__':
    y_train = datasets.MNIST('../data', train=True).train_labels.numpy()
    y_test = datasets.MNIST('../data', train=False).test_labels.numpy()

    cnn = CNN(epochs=10)

    mod_val = 1
    train_idx = torch.np.arange(MNIST_TRAIN_SIZE)[torch.np.arange(MNIST_TRAIN_SIZE) % mod_val == 0]
    train_labels = y_train[torch.np.arange(MNIST_TRAIN_SIZE) % mod_val == 0]
    sample_weight = torch.np.ones(len(train_labels)) /len(train_labels)
    sample_weight[(train_labels == 9) | (train_labels == 8)] *= 5
    sample_weight = sample_weight / sum(sample_weight)

    cnn.fit(train_idx, train_labels, sample_weight)
    # cnn.model = m
    # m = cnn.model = m

    # Test to make sure predict is working
    assert((cnn.predict_proba(loader='test').argmax(axis=1) == cnn.predict(loader='test')).all())

    cnn.test()
    pred = cnn.predict(loader='test')
    torch.np.bincount(y_test[pred == y_test]) / torch.np.bincount(y_test).astype(float)

