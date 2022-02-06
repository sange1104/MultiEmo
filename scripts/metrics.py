import torch
import pandas as pd

def single_score(pred, actual, k):
    '''Calculates the single accuracy for top-k predictions'''
    top_k = []
    for i in range(len(pred)):
        top_k.append(torch.argsort(pred[i])[-k:].tolist())
    acc = len([i for i in range(len(pred)) if actual[i] in top_k[i]])
    return acc / len(pred)

def category_score(pred, actual, df, k): 
    '''Calculates the category-level accuracy for top-k predictions'''
    #get top_k label
    top_k = []
    for i in range(len(pred)):
        top_k.append(torch.argsort(pred[i])[-k:].tolist())
     
    #make emoji_label : category label, sentiment label dictionary
    info = {}
    for i in range(len(df)):
        single, cat, sent = df.single[i], df.cat[i], df.sent[i]
        info[single] = {'cat' : cat, 'sent' : sent}
    
    #get number of correct samples
    correct = 0
    for i, pred in enumerate(top_k):
        pred_cat = []
        for l in pred:
            pred_cat.append(info[l]['cat'])
        if info[actual[i]]['cat'] in pred_cat:
            correct += 1
    
    return correct / len(actual)

def sentiment_score(pred, actual, df): 
    '''Calculates the sentiment-level accuracy for top-k predictions'''
    # get top_k label
    label = []
    for i in range(len(pred)):
        label.append(torch.argsort(pred[i])[-1:].tolist())  
     
    #make emoji_label : category label, sentiment label dictionary
    info = {}
    for i in range(len(df)):
        single, cat, sent = df.single[i], df.cat[i], df.sent[i]
        info[single] = {'cat' : cat, 'sent' : sent}
    
    #get number of correct samples
    correct = 0
    for i, pred in enumerate(label):
        pred_sent = [] 
        for l in pred:
            pred_sent.append(info[l]['sent'])
        if info[actual[i]]['sent'] in pred_sent:
            correct += 1
     
    
    return correct / len(actual)

def print_metric(pred, true, name, k):
    print('-'*50) 
    print('%s'%name)
    print('-'*50) 
    if name == 'Sentiment score':
        score = sentiment_score(pred, true, df_metric) 
        print('Accuracy : %.4f'%score)
    else:
        for i in range(k):
            if name == 'Single score':
                score = single_score(pred, true, i+1) 
            elif name == 'Category score':
                score = category_score(pred, true, df_metric, i+1)  
            print('Top %d accuracy : %.4f'%(i+1, score))
    print()

# load dataframe which maps each emoji and corresponding category and sentiment label
df_metric = pd.read_csv("./data/emoji_map.csv", index_col=0, header=0)   
