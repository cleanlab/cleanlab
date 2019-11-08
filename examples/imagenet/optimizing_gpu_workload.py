
# coding: utf-8

# # Optimize training ImageNet by distributing jobs across 4 GPUs as evenly as possible
# 
# ### This is not necessary for the experiments in our paper, it just speeds things up a bit.

# In[128]:


import numpy as np
import sys


# In[129]:


jobs = {'resnet18_00': 0,
     
#  'resnet18_100_argmax': 406060,
 'resnet18_100_both': 208828,
 'resnet18_100_cj_only': 140693,
 'resnet18_100_cl_pbc': 208828,
 'resnet18_100_cl_pbnr': 194125,
 'resnet18_100random': 178294,
     
 'resnet18_20_argmax': 81212,
 'resnet18_20_both': 41765,
 'resnet18_20_cj_only': 28138,
 'resnet18_20_cl_pbc': 41765,
 'resnet18_20_cl_pbnr': 38825,
 'resnet18_20random': 35658,
     
 'resnet18_40_argmax': 162424,
 'resnet18_40_both': 83531,
 'resnet18_40_cj_only': 56277,
 'resnet18_40_cl_pbc': 83531,
 'resnet18_40_cl_pbnr': 77650,
 'resnet18_40random': 71317,
     
#  'resnet18_60_argmax': 243636,
 'resnet18_60_both': 125296,
 'resnet18_60_cj_only': 84415,
 'resnet18_60_cl_pbc': 125296,
 'resnet18_60_cl_pbnr': 116475,
 'resnet18_60random': 106976,
     
#  'resnet18_80_argmax': 324848,
 'resnet18_80_both': 167062,
 'resnet18_80_cj_only': 112554,
 'resnet18_80_cl_pbc': 167062,
 'resnet18_80_cl_pbnr': 155300,
 'resnet18_80random': 142635,
}
j = {v:k for k,v in jobs.items()}


# In[130]:


# Stochastically searching for a balanced way distribution of workload
# Kill once you think the (max - min) is low enough. Something around 7000 is good.
best_rands = None
best_score = np.inf
while(True):
    rands = np.random.rand(len(jobs))
    scores = [sum((1281167 - np.array(list(jobs.values())))[(rands < i) & (rands >= (i - 0.25))]) for i in [0.25, 0.5, 0.75, 1.]]
    score = np.max(scores) - np.min(scores)
    if score < best_score:
        best_score = score
        best_rands = rands
        print(score)
        sys.stdout.flush()


# In[131]:


partitions = [list(np.array(list(jobs.keys()))[(best_rands < i) & (best_rands >= (i - 0.25))]) for i in [0.25, 0.5, 0.75, 1.]]

# Verify partitions are reasonable
[np.sum([jobs[k] for k in p]) for p in partitions]


# In[134]:


# Generate jobs from partitions
b = 156  # batch size
for trial in [2,3,4,5]:
    for g, p in enumerate(partitions):
        print('\nGPU: {}\n'.format(g))
        for f in p:
            print('mkdir /home/cgn/masked_imagenet_training/resnet18/trial{}/{}'.format(trial, f))
            print('cd /home/cgn/masked_imagenet_training/resnet18/trial{}/{}'.format(trial, f))
            amt = f[9:].split('_')[0]
            method = f[10 + len(amt):]
            py = 'python3 /home/cgn/cgn/cleanlab/examples/imagenet/imagenet_train_crossval.py'
            params = ' -a "resnet18" --lr 0.1 -b {} --gpu {} '.format(b,g)
            mask = '-m /home/cgn/masks/imagenet_train_bool_{}_mask__fraction_{}.npy '.format(method, amt)
            suffix = '/datasets/datasets/imagenet/ >> out.log'
            cmd = py + params + mask + suffix
            print(cmd)


# # Now we do the same thing but we take advantage of the fact that with batch size 156 we can run two models on one GPU at a time. Its slower per model, but faster overall.

# In[127]:


# Stochastically searching for a balanced way distribution of workload
# Kill once you the max and min scores are nearest too each other.
# This is a much harder task because the number of bins is significantly higher.
# A decent value might be 70,000.
best_rands = None
best_score = np.inf
while(True):
    rands = np.random.rand(len(jobs))
    scores = [sum((1281167 - np.array(list(jobs.values())))[(rands < i) & (rands >= (i - 0.125))]) for i in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.]]
    score = max(scores) - min(scores)
    if score < best_score:
        best_score = score
        best_rands = rands
        print(score)
        sys.stdout.flush()


# In[113]:


partitions = [list(np.array(list(jobs.keys()))[(best_rands < i) & (best_rands >= (i - 0.125))]) for i in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.]]

# Verify partitions are reasonable
[np.sum([jobs[k] for k in p]) for p in partitions]


# In[115]:


partitions


# In[114]:


# Generate jobs from partitions
for g, p in enumerate(partitions):
    print('\nGPU: {}\n'.format(g // 2))
    for f in p:
        print('mkdir /home/cgn/masked_imagenet_training/resnet18/{}'.format(f))
        print('cd /home/cgn/masked_imagenet_training/resnet18/{}'.format(f))
        amt = f[9:].split('_')[0]
        method = f[10 + len(amt):]
        py = 'python3 /home/cgn/cgn/cleanlab/examples/imagenet/imagenet_train_crossval.py'
        params = ' -a "resnet18" --lr 0.1 -b 256 --gpu {} '.format(g // 2)
        mask = '-m /home/cgn/masks/imagenet_train_bool_{}_mask__fraction_{}.npy '.format(method, amt)
        suffix = '/datasets/datasets/imagenet/ >> out.log'
        cmd = py + params + mask + suffix
        print(cmd)

