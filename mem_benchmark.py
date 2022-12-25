from cleanlab.filter_copy import find_label_issues_bench, find_label_issues_cow
#from old_filter_code import get_noise_indices
import numpy as np
np.set_printoptions(suppress=True)
import multiprocessing
from time import time

def normalize(arr):
    normalized = np.zeros(arr.shape, dtype=np.float16)
    for i, a in enumerate(arr):
        normalized[i] = a / np.sum(a)
    return normalized

m = 5000
n = 10000

nt = 9
ncores = 4
time_ints = np.zeros((9, 10))

pred_probs = np.random.randint(low=1, high=100, size=[n, m], dtype=np.uint8)
pred_probs = normalize(pred_probs)
labels = np.repeat(np.arange(m), n // m)

if __name__=='__main__':
    start_time = time()
    multiprocessing.set_start_method('fork')
    for n_jobs in range(ncores):
        print(f"starting job {n_jobs}")
        _, times = find_label_issues_cow(pred_probs=pred_probs, labels=labels, n_jobs=n_jobs+1)
        #_ = get_noise_indices(psx=pred_probs, s=labels, prune_method='prune_by_noise_rate', n_jobs=n_jobs+1)
        print(f"finish job {n_jobs}")
    #print(f"total time: {time() - start_time}")
