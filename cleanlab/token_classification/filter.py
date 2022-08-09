import numpy as np 
from cleanlab.filter import find_label_issues as find_label_issues_main 

def find_label_issues(labels, pred_probs, return_indices_ranked_by='self_confidence'): 
    labels_flatten = [l for label in labels for l in label] 
    pred_probs_flatten = [pred for pred_prob in pred_probs for pred in pred_prob] 
    pred_probs_flatten = np.array(pred_probs_flatten) 
    
    issues = find_label_issues_main(labels_flatten, pred_probs_flatten, return_indices_ranked_by=return_indices_ranked_by) 
    
    lengths = [len(label) for label in labels] 
    mapping = [[(i, j) for j in range(length)] for i, length in enumerate(lengths)] 
    mapping = [index for indicies in mapping for index in indicies] 
    
    issues = [mapping[issue] for issue in issues] 
    return issues 
