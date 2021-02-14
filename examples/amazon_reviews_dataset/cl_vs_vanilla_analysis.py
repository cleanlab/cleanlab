#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Analyze results from compare_cl_vs_vanilla


# In[1]:


import os
import pandas as pd


# In[75]:


base = '/home/cgn/amazon_reviews/'
results = []
for seed in range(5):
    for trainsize in [500000, 1000000]:
        for epochs in [5, 20, 50]:
            if trainsize == 500000 and epochs == 50:
                continue
            fn = 'out_seed_{}_trainsize_{}_epochs_{}.log'.format(
                    seed, trainsize, epochs)
#             print(fn)
            with open(base + fn, 'r') as f:
                result = f.readlines()
                lod = eval(result[-5:-4][0].strip())
                settings = {'seed': seed,
                            'trainsize': trainsize,
                            'epochs': epochs}
                [d.update(settings) for d in lod]
                results += lod


# In[76]:


data = pd.DataFrame(results)
data = data[data['method'] != 'cl_intersection_all_methods']
data.drop('train_size', axis=1, inplace=True)
data['trainsize'] = (data['trainsize'] / 1000).round().astype(int).astype(str) + 'K'


# In[83]:


cols = ['Top-1 Acc', 'Pruned']
df = data.groupby(['test_split', 'trainsize', 'epochs', 'method']).agg(['mean', 'std'])
# df.columns = [' '.join(col).strip() for col in df.columns.values]
df.drop([('seed', 'mean'), ('seed', 'std')], axis=1, inplace=True)
# Formatting
df[('data_removed', 'mean')] = (df[('data_removed', 'mean')] / 1000).round().astype(int).astype(str) + 'K'
# df[('data_removed', 'std')] = df[('data_removed', 'std')].round().astype(int)
del df[('data_removed', 'std')]
df[('acc', 'mean')] = (df[('acc', 'mean')] * 100).round(1)
df[('acc', 'std')] = (df[('acc', 'std')] * 100).round(2)
df.columns = [' '.join(col).strip() for col in df.columns.values]
df['acc'] = df['acc mean'].astype(str) + '±' + df['acc std'].astype(str)
del df['acc mean']
del df['acc std']
df = df[['acc', 'data_removed mean']]
df.columns = cols
# pd.concat([z for i, z in df.reset_index().groupby(['test_split', 'trainsize'])])
sdfs = []
for key, sdf in df.reset_index(level=[1,2]).groupby(['trainsize', 'epochs']):
    z = sdf[cols]
    label1 = r'$N = {}$'.format(key[0])
    label2 = 'epochs = {}'.format(key[1])
    z.columns = pd.MultiIndex.from_product([[label1], [label2], z.columns])
    sdfs.append(z)
paper_df = pd.concat(sdfs, axis=1)
for c in [('$N = 1000K$',  'epochs = 5',    'Pruned'),
          ('$N = 1000K$', 'epochs = 20',    'Pruned'),
          ('$N = 500K$',  'epochs = 5',     'Pruned'),]:
    del paper_df[c]
paper_df


# In[84]:


method_name_map = {
	'argmax': r'CL: $\bm{C}_{\text{confusion}}$',
	'pbc': 'CL: PBC',
	'cj\_only': r'CL: $\cj$',
	'both': 'CL: C+NR',
	'pbnr': 'CL: PBNR',
	'vanilla': 'Baseline',
}


# In[81]:


tex = paper_df.to_latex().replace('\\$', '$').replace('±', '$\pm$')
tex = ' '.join([method_name_map. get(i, i) for i in tex.split(' ')])
print(tex)


# In[ ]:




