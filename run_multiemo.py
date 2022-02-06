'''This file runs simple demo for emoji prediction'''
from scripts.model import MultiEmo, TorchMoji, load_specific_weights
from scripts.load_data import InferDataset, SentenceTokenizer
import argparse
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__': 
    ###################
    # Configuration
    ###################
    parser = argparse.ArgumentParser() 
    parser.add_argument('--aux_num', type=int, required=True)
    parser.add_argument('--aux_task', type=str, required=True)
    args = parser.parse_args()

    MAIN_IDX = 0
    AUX_NUM = args.aux_num
    AUX_TASK = args.aux_task.split(' ')
    assert len(AUX_TASK)==AUX_NUM, 'Does not match the number of auxiliary task'
    for aux in AUX_TASK:
        if aux not in ['emo', 'Ekman', 'sent']:
            raise AssertionError('Invalid format for auxiliary task. Argument should be one of "emo", "Ekman", and "sent".')

    #########################
    # Get text from an user
    #########################
    print()
    print('-----------------------------------------------------------------------------------------')
    print('Simple Demo for emoji prediction!')
    print('Type a sentence, then our model will predict the most relevant 5 emojis for your text.')
    print('-----------------------------------------------------------------------------------------')
    text = input('Type: ')
    st = SentenceTokenizer(len(text.split(' '))) 
    data = InferDataset(text = [text], 
                        tokenizer = st) 

    ###################
    # load model
    ###################
    deepmoji_dim = 2304
    torchmoji_weight_path = './data/pytorch_model.bin'
    torchmoji = TorchMoji() 
    load_specific_weights(torchmoji, torchmoji_weight_path, exclude_names=['output_layer'])
    model = MultiEmo(AUX_TASK, torchmoji, deepmoji_dim).cuda()
    multiemo_weight_path = './checkpoints/model_aux_%s.pt'%('_'.join(AUX_TASK)) # **input your weight path!!
    model.load_state_dict(torch.load(multiemo_weight_path))

    ###################
    # get prediction
    ###################
    model.eval()

    with torch.no_grad():
        # load data    
        assert len(data) == 1
        for batch in data:
            tokens = batch["token"]
        
            # model forward
            outputs = model(
                tokens = torch.LongTensor(tokens).unsqueeze(0).cuda(),
                tasks = torch.zeros(1, 1).cuda()
                ) 

    #########################
    # get top-5 prediction
    #########################
    k = 5 
    top_k_pred = torch.argsort(outputs[MAIN_IDX][0])[-k:].tolist()  
    
    ###################
    # print result
    ################### 
    # load dataframe which maps int label to each emoji 
    df_metric = pd.read_csv("./data/emoji_info_64.csv", index_col=0, header=0)   
    top_k_emoji = [df_metric[df_metric.index==i].emoji.item() for i in top_k_pred]
    print('Top-5 emoji is %s, %s, %s, %s, %s'%tuple(top_k_emoji)) 
    print()