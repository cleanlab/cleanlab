"""
Local test script for datalab.py
"""

import numpy as np
import pandas as pd
from datasets import load_dataset

from cleanlab import Datalab

# pandas dataset:  my_data = pd.read_csv(...)
my_data = load_dataset("lhoestq/demo1", split="train")  # label column is 'star'
n = len(my_data)
k = 3


def remap_labels(example):
    example["star"] = example["star"] - k
    return example


my_data = my_data.map(remap_labels)  # make labels start from 0
my_df = pd.DataFrame(my_data)  # try to avoid using this except where we want to return DF objects

my_data.info


pred_probs = np.random.rand(n, k)
pred_probs = pred_probs / pred_probs.sum(axis=1)[:, np.newaxis]

datalab = Datalab(my_data, label_name="star")
datalab.find_issues(pred_probs=pred_probs)

print(datalab)

print(datalab.issues)

path = "temp_datalab/"
datalab.save(path)

datalab = None
datalab = Datalab.load(path, data=my_data)

print(datalab.issues)
