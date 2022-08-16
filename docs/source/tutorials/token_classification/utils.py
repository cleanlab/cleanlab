import string 
import numpy as np 
import os 
import pandas as pd 
from termcolor import colored 

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

def modified(given_words, sentence_tokens): 
    for word, token in zip(given_words, sentence_tokens): 
        if ''.join(word) != ''.join(token): 
            return True  
    return False 

def process_token(token, replace=[('#', '')]): 
    for old, new in replace: 
        token = token.replace(old, new) 
    return token 

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

def color_sentence(sentence, word): 
    start_idx = sentence.index(word) 
    before, after = sentence[:start_idx], sentence[start_idx + len(word):]
    return '%s%s%s' % (before, colored(word, 'red'), after) 


def display_issues(issues, given_words, *, 
                   pred_probs=None, 
                   given_labels=None, 
                   exclude=[], 
                   class_names=None, 
                   top=20): 
    shown = 0 
    is_tuple = type(issues[0]) == tuple  
    
    for issue in issues: 
        if is_tuple: 
            i, j = issue 
            sentence = get_sentence(given_words[i]) 
            word = given_words[i][j] 
            
            if pred_probs: 
                prediction = pred_probs[i][j].argmax() 
                if class_names: 
                    prediction = class_names[prediction] 
            if given_labels: 
                given = given_labels[i][j] 
                if class_names: 
                    given = class_names[given] 
            if pred_probs and given_labels: 
                if (given, prediction) in exclude: 
                    continue 
            
            shown += 1 
            print('Sentence %d, token %d: \n%s' % (i, j, color_sentence(sentence, word))) 
            if given_labels and not pred_probs: 
                print('Given label: %s\n' % str(given)) 
            elif not given_labels and pred_probs: 
                print('Predicted label: %s\n' % str(prediction)) 
            elif given_labels and pred_probs: 
                print('Given label: %s, predicted label: %s\n' % (str(given), str(prediction))) 
            else: 
                print() 
        else: 
            shown += 1 
            sentence = get_sentence(given_words[issue]) 
            print('Sentence %d: %s\n' % (issue, sentence)) 
        if shown == top: 
            break 

def common_label_issues(issues, given_words, *, 
                        labels=None, 
                        pred_probs=None, 
                        class_names=None, 
                        top=10, 
                        exclude=[], 
                        verbose=True): 
    count = {} 
    if not labels or not pred_probs: 
        for issue in issues: 
            i, j = issue 
            word = given_words[i][j] 
            if word not in count: 
                count[word] = 0 
            count[word] += 1 
            
        words = [word for word in count.keys()] 
        freq = [count[word] for word in words] 
        rank = np.argsort(freq)[::-1][:top] 
        
        for r in rank: 
            print("Token '%s' is potentially mislabeled %d times throughout the dataset\n" % (words[r], freq[r])) 
            
        info = [[word, f] for word, f in zip(words, freq)] 
        info = sorted(info, key=lambda x: x[1], reverse=True) 
        return pd.DataFrame(info, columns=['token', 'num_label_issues']) 
    
    if not class_names: 
        print("Classes will be printed in terms of their integer index since `class_names` was not provided. ") 
        print("Specify this argument to see the string names of each class. \n") 
        
    n = pred_probs[0].shape[1] 
    for issue in issues: 
        i, j = issue 
        word = given_words[i][j] 
        label = labels[i][j] 
        pred = pred_probs[i][j].argmax() 
        if word not in count: 
            count[word] = np.zeros([n, n], dtype=int) 
        if (label, pred) not in exclude: 
            count[word][label][pred] += 1 
    words = [word for word in count.keys()] 
    freq = [np.sum(count[word]) for word in words] 
    rank = np.argsort(freq)[::-1][:top] 
    
    for r in rank: 
        matrix = count[words[r]] 
        most_frequent = np.argsort(count[words[r]].flatten())[::-1] 
        print("Token '%s' is potentially mislabeled %d times throughout the dataset" % (words[r], freq[r])) 
        if verbose: 
            print('---------------------------------------------------------------------------------------') 
            for f in most_frequent: 
                i, j = f // n, f % n 
                if matrix[i][j] == 0: 
                    break 
                if class_names: 
                    print('labeled as class `%s` but predicted to actually be class `%s` %d times' % (class_names[i], class_names[j], matrix[i][j])) 
                else: 
                    print('labeled as class %d but predicted to actually be class %d %d times' % (i, j, matrix[i][j])) 
        print() 
    info = [] 
    for word in words: 
        for i in range(n): 
            for j in range(n): 
                num = count[word][i][j] 
                if num > 0: 
                    if not class_names: 
                        info.append([word, i, j, num]) 
                    else: 
                        info.append([word, class_names[i], class_names[j], num]) 
    info = sorted(info, key=lambda x: x[3], reverse=True) 
    return pd.DataFrame(info, columns=['token', 'given_label', 'predicted_label', 'num_label_issues']) 
    

def filter_by_token(token, issues, given_words): 
    returned_issues = [] 
    for issue in issues: 
        i, j = issue 
        if token.lower() == given_words[i][j].lower(): 
            returned_issues.append(issue) 
    return returned_issues 


def issues_from_scores(sentence_scores, token_scores, threshold=0.2): 
    ranking = np.argsort(sentence_scores) 
    cutoff = 0 
    while sentence_scores[ranking[cutoff]] < threshold and cutoff < len(ranking): 
        cutoff += 1 
    ranking = ranking[:cutoff] 
    if not token_scores: 
        return list(ranking) 
    else: 
        return [(r, token_scores[r].argmin()) for r in ranking] 
    