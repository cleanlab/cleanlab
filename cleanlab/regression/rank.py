import numpy as np 

def get_label_quality_score(
    true_labels: np.ndarray,
    pred_labels: np.ndarray
)-> np.ndarray:
    """
    Returns label quality score
    
    Score is continous value in range [0,1]
    
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).
    """
    residual = true_labels - pred_labels
    quality_scores = np.exp(-abs(residual))
    return quality_scores


if __name__ == "__main__":
## WILL BE DELETED LATER 
    a = np.array([1,2,3,4])
    b = np.array([2,2,5,4.1])
    print(get_label_quality_score(a,b))
