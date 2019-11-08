
# coding: utf-8

# In[1]:


from cleanlab import noise_generation
import torchvision
from torchvision import transforms
import os
import sys
import numpy as np
import json
import pickle


# In[2]:


# # Prepare CIFAR-10 dataset for PyTorch dataloader
# import os
# for kind in ['test/', 'train/']:
#     data_path = '/datasets/datasets/cifar10/cifar10/'
#     for file in os.listdir(data_path + kind):
#         class_name = file.split('_')[-1].split('.')[0]
#         os.system('mv {} {}'.format(data_path+kind+file, data_path+kind+class_name+"/"+file))


# In[3]:


# Create json with train labels
for cifar_dataset in ["cifar10", "cifar100"]:
    data_path = '/datasets/datasets/{}/{}/'.format(cifar_dataset, cifar_dataset)
    train_dataset = torchvision.datasets.ImageFolder(data_path + 'train/')
    d = dict(train_dataset.imgs)
    # Store the dictionary        
    with open(data_path + "train_filename2label.json", 'w') as wf:
        wf.write(json.dumps(d, indent=4))


# In[32]:


# Create json with test labels
for cifar_dataset in ["cifar10", "cifar100"]:
    data_path = '/datasets/datasets/{}/{}/'.format(cifar_dataset, cifar_dataset)
    test_dataset = torchvision.datasets.ImageFolder(data_path + 'test/')
    d = dict(test_dataset.imgs)
    # Store the dictionary        
    with open(data_path + "test_filename2label.json", 'w') as wf:
        wf.write(json.dumps(d, indent=4))


# In[4]:


# Create noisy labels for both CIFAR-10 and CIFAR-100
for cifar_dataset in ["cifar10", "cifar100"]:
    data_path = '/datasets/datasets/{}/{}/'.format(cifar_dataset, cifar_dataset)
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path + 'train/',
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )
    y = train_dataset.targets
    K = int(cifar_dataset[5:])
    print(cifar_dataset, np.bincount(y))
    for frac_zero_noise_rates in np.arange(0, 0.8, 0.2):
        for noise_amount in np.arange(0, 1, 0.2):
            print('noise_amount', round(noise_amount, 1), '| frac_zero_noise_rates', round(frac_zero_noise_rates, 1))

            # Generate class-conditional noise        
            nm = noise_generation.generate_noise_matrix_from_trace(
                K=K,
                trace=int(K * (1 - noise_amount)),
                valid_noise_matrix=False,
                frac_zero_noise_rates=frac_zero_noise_rates,
                seed=0,
            )

            # noise matrix is valid if diagonal maximizes row and column
            valid = all((nm.argmax(axis=0) == range(K)) & (nm.argmax(axis=1) == range(K)))
            print('valid:', valid)

            # Create noisy labels
            np.random.seed(seed=0)
            s = noise_generation.generate_noisy_labels(y, nm)
            
            # Check accuracy of s and y
            print('Accuracy of s and y:', sum(s==y)/len(s))

            # Create map of filenames to noisy labels
            d = dict(zip([i for i,j in train_dataset.imgs], [int(i) for i in s]))

            # Store dictionary as json
            wfn_base = '{}_noisy_labels__frac_zero_noise_rates__{}__noise_amount__{}'.format(
                cifar_dataset,
                "0.0" if frac_zero_noise_rates  < 1e-4 else round(frac_zero_noise_rates, 1),
                "0.0" if noise_amount < 1e-4 else round(noise_amount, 1),
            )
            wfn = data_path + "noisy_labels/" + wfn_base
            print(wfn)

            # Store the dictionary        
            with open(wfn + ".json", 'w') as wf:
                wf.write(json.dumps(d))

            # Store the noise matrix as well
            wfn_base = "{}_noise_matrix".format(cifar_dataset) + "__" + "__".join(wfn_base.split("__")[1:])
            wfn = data_path + "noisy_labels/" + wfn_base
            print(wfn)
            with open(wfn + ".pickle", 'wb') as wf:
                pickle.dump(nm, wf, protocol=pickle.HIGHEST_PROTOCOL)


# # View the noise matrices

# In[17]:


# Create noisy labels for both CIFAR-10 and CIFAR-100
# Store dictionary as json
import numpy as np
import pickle
from cleanlab import util
for cifar_dataset in ["cifar10"]:  #, "cifar100"]:
    data_path = '/datasets/datasets/{}/{}/'.format(cifar_dataset, cifar_dataset)
    for noise_amount in np.arange(0.2, 0.61, 0.2):
        for frac_zero_noise_rates in np.arange(0, 0.61, 0.2):
            # Print the noise matrix
            rfn_base = '{}_noisy_labels__frac_zero_noise_rates__{}__noise_amount__{}'.format(
                cifar_dataset,
                "0.0" if frac_zero_noise_rates  < 1e-4 else round(frac_zero_noise_rates, 1),
                "0.0" if noise_amount < 1e-4 else round(noise_amount, 1),
            )
            rfn = data_path + "noisy_labels/" + rfn_base
            rfn_base = "{}_noise_matrix".format(cifar_dataset) + "__" + "__".join(rfn_base.split("__")[1:])
            rfn = data_path + "noisy_labels/" + rfn_base
            with open(rfn + ".pickle", 'rb') as rf:
                nm = pickle.load(rf)
            actual_noise = 0.7 if abs(noise_amount - 0.6) < 1e-3 else noise_amount
            print('Noise amount:', round(actual_noise, 3), "| Sparsity:", round(frac_zero_noise_rates, 3))
            util.print_noise_matrix(nm)

