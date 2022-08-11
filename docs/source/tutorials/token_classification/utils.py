import string 
import numpy as np 
import os 

def get_sentence(words): 
    sentence = ''
    for word in words:
        if word not in string.punctuation or word in ['-', '(']:
            word = ' ' + word
        sentence += word
    sentence = sentence.replace(" '", "'").replace('( ', '(').strip()
    return sentence

def filter_sentence(sentences, condition=None, return_mask=True): 
    if not condition: 
        condition = lambda sentence: len(sentence) > 1 and '#' not in sentence 
    mask = list(map(condition, sentences)) 
    sentences = [sentence for m, sentence in zip(mask, sentences) if m] 
    if return_mask: 
        return sentences, mask 
    else: 
        return sentences 
    
    
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

def mapping(entities, maps): 
    f = lambda x: maps[x] 
    return list(map(f, entities)) 

def merge_probs(probs, maps): 
    old_classes = probs.shape[1] 
    probs_merged = np.zeros([len(probs), np.max(maps)+1]) 
    
    for i in range(old_classes): 
        if maps[i] >= 0: 
            probs_merged[:, maps[i]] += probs[:, i] 
    if -1 in maps: 
        row_sums = probs_merged.sum(axis=1) 
        probs_merged /= row_sums[:, np.newaxis] 
    return probs_merged 

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

def get_pred_probs_and_labels(scores, tokens, given_token, given_label, weighted=False): 
    i, j = 0, 0 
    pred_probs, labels = [], [] 
    for token, label in zip(given_token, given_label): 
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
        labels.append(label)
        
    return np.array(pred_probs), labels 

def to_dict(nl): 
    return {str(i): l for i, l in enumerate(nl)} 

def read_npz(filepath): 
    data = dict(np.load(filepath)) 
    data = [data[str(i)] for i in range(len(data))] 
    return data 

def frequent_words(issues, words, labels, pred_probs, exclude=[]): 
    count = {} 
    n = pred_probs.shape[1] 
    predictions = np.argmax(pred_probs, axis=1) 
    for issue in issues: 
        word, label, pred = words[issue], labels[issue], predictions[issue] 
        if word not in count: 
            count[word] = np.zeros([n, n], dtype=int) 
        if (label, pred) not in exclude: 
            count[word][label][pred] += 1 
    return count 

def show_frequent_issues(count, entities, top=10, verbose=False): 
    words = [word for word in count.keys()] 
    n = count[words[0]].shape[0] 
    freq = [np.sum(count[word]) for word in words] 
    rank = np.argsort(freq)[::-1][:10] 
    
    for r in rank: 
        word = words[r] 
        matrix = count[word] 
        most_frequent = np.argsort(matrix.flatten())[::-1] 
        print("'%s' is mislabeled %d times" % (word, freq[r])) 
        if verbose: 
            print('-----------------------------') 
            for f in most_frequent: 
                i, j = f // n, f % n 
                if matrix[i][j] == 0: 
                    break 
                print('labeled as %s, but predicted as %s %d times' % 
                      (entities[i], entities[j], matrix[i][j])) 
        print() 
        
def search_token(token, issues, mapping, words): 
    indicies = [] 
    for issue in issues: 
        i, j = mapping[issue] 
        if words[i][j] == token: 
            indicies.append(i) 
    indicies = list(set(indicies)) 
    indicies.sort() 
    return indicies 