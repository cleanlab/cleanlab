import numpy as np 
import os 

def create_folds(sentences, k=10, path='folds/', seed=0): 
    np.random.seed(seed) 
    indicies = [i for i in range(len(sentences))] 
    np.random.shuffle(indicies) 
    partitions = [[sentences[index] for index in indicies[i::k]] for i in range(k)] 
    
    if os.path.exists(path): 
        print("'%s' already exists, skipping..." % path) 
    else: 
        os.system('mkdir folds') 
        for i in range(k): 
            os.system('mkdir folds/fold%d' % i) 

        def write_sentences_to_file(sentences, path): 
            for sentence in sentences: 
                for line in sentence: 
                    path.write(line) 
                path.write('\n')     

        for i in range(k): 
            train = open('folds/fold%d/train.txt' % i, "a") 
            test = open('folds/fold%d/test.txt' % i, "a") 
            for j in range(k): 
                write_sentences_to_file(partitions[j], test if i == j else train) 
            train.close() 
            test.close() 
        
    indicies = [[index for index in indicies[i::k]] for i in range(k)] 
    return indicies 


def modified(given_words, sentence_tokens): 
    for word, token in zip(given_words, sentence_tokens): 
        if ''.join(word) != ''.join(token): 
            return True  
    return False 


def get_probs(sentence, pipe, maps=None): 
    def softmax(logit): 
        return np.exp(logit) / np.sum(np.exp(logit)) 
    
    forward = pipe.forward(pipe.preprocess(sentence)) 
    logits = forward['logits'][0].numpy() 
    probs = np.array([softmax(logit) for logit in logits]) 
    probs = probs[1:-1] 
    
    if not maps: 
        return probs 
    return merge_pros(probs, maps) 


def get_pred_probs(scores, tokens, given_token, weighted=False): 
    i, j = 0, 0 
    pred_probs = [] 
    for token in given_token: 
        i_new, j_new = i, j 
        acc = 0 
        
        weights = []         
        while acc != len(token): 
            token_len = len(tokens[i_new][j_new:]) 
            remain = len(token) - acc 
            weights.append(min(remain, token_len)) 
            if token_len > remain: 
                acc += remain 
                j_new += remain 
            else: 
                acc += token_len 
                i_new += 1 
                j_new = 0 
        
        if i != i_new: 
            probs = np.average(scores[i:i_new], axis=0, weights=weights if weighted else None) 
        else: 
            probs = scores[i] 
        i, j = i_new, j_new 
        
        pred_probs.append(probs) 
        
    return np.array(pred_probs) 


def to_dict(nl): 
    return {str(i): l for i, l in enumerate(nl)} 


def read_npz(filepath): 
    data = dict(np.load(filepath)) 
    data = [data[str(i)] for i in range(len(data))] 
    return data 