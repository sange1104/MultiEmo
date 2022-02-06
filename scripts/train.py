from model import MultiEmo, TorchMoji, load_specific_weights
from load_data import create_data_loader, SentenceTokenizer
from infer import get_output
from metrics import print_metric, single_score, category_score, sentiment_score
from utils import save_history, print_result, lr_decay, save_checkpoint
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, aux_task, data_loader, loss_fn, optimizer):
    '''
    # Arguments
      - model: MultiEmo nn for training
      - aux_task: a list of auxiliary task
      - data_loader: data loader for training set
      - loss_fn: torch loss function to calculate loss
      - optimizer: torch optimizer to update params in the model
    ''' 
    model = model.train()
    
    # initialize history list - loss, accuracy
    torch_loss_dict = defaultdict()
    loss_dict = defaultdict(float)
    acc_dict = defaultdict(float)
    numdata_dict = defaultdict(int) 

    for batch in tqdm(data_loader):  
        # load data from train data loader
        task = batch["task"].to(device)
        tokens = batch["token"].to(device) 
        targets = batch["label"].to(device)  
        
        # set target dictionary
        target_dict = {}
        target_dict[MAIN_IDX] = targets[task==MAIN_IDX]
        for aux in aux_task:
            idx = TASK_TO_IDX[aux]
            target_dict[idx] = targets[task==idx] 
        
        # get the number of data for each task 
        numdata_dict[MAIN_IDX] += len(target_dict[MAIN_IDX]) 
        for aux in aux_task:
            idx = TASK_TO_IDX[aux]
            numdata_dict[idx] += len(target_dict[idx]) 
            
        # initialize optimizer
        optimizer.zero_grad()
    
        # model forward 
        output_dict = model(
          tokens=tokens,
            tasks=task
        )  
        
        loss = []
        #######################
        # Main task
        #######################
        # loss update with emoji prediction task
        outputs = output_dict[MAIN_IDX]
        targets = target_dict[MAIN_IDX] 
        if len(outputs) > 0: 
            loss_main = loss_fn(outputs, targets) 
            loss.append(loss_main)
            loss_dict[MAIN_IDX] += loss_main.item() 

            # number of correct prediction
            _, pred_int = torch.max(outputs, dim=1)  
            acc_dict[MAIN_IDX] += torch.sum(pred_int == targets).item()

        
        #######################
        # Auxiliary task
        #######################
        for aux in aux_task:
            idx = TASK_TO_IDX[aux]
            outputs = output_dict[idx]
            targets = target_dict[idx] 
            if len(outputs) > 0:
                # loss update with auxiliary task
                loss_aux = loss_fn(outputs, targets) 
                loss.append(loss_aux)
                torch_loss_dict[idx] = loss_aux
                loss_dict[idx] += loss_aux.item()  

                # number of correct prediction
                _, pred_int = torch.max(outputs, dim=1) 
                acc_dict[idx] += torch.sum(pred_int == targets).item()
         
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # calculate sum of loss
        total_loss = sum(loss)
        total_loss.backward()
        # update parameters
        optimizer.step()  

    # calculate accuracy and average loss 
    acc_dict[MAIN_IDX] /= numdata_dict[MAIN_IDX]
    loss_dict[MAIN_IDX] /= numdata_dict[MAIN_IDX]
    
    for aux in aux_task: 
        idx = TASK_TO_IDX[aux]
        acc_dict[idx] /= numdata_dict[idx]
        loss_dict[idx] /= numdata_dict[idx]  

    return model, acc_dict, loss_dict

def eval_model(model, aux_task, data_loader, loss_fn):
    '''
    # Arguments 
      - model: MultiEmo nn for evaluation
      - aux_task: a list of auxiliary task
      - data_loader: data loader for validation set
      - loss_fn: torch loss function to calculate loss 
    ''' 
    model = model.eval()

    # initialize history list - loss, accuracy
    torch_loss_dict = defaultdict()
    loss_dict = defaultdict(float)
    acc_dict = defaultdict(float)
    numdata_dict = defaultdict(int) 
 
    with torch.no_grad():
        for batch in tqdm(data_loader): 
            # load data from val data loader
            task = batch['task'].to(device)
            tokens = batch["token"].to(device) 
            targets = batch["label"].to(device)
            
            # set target dictionary
            target_dict = {}
            target_dict[MAIN_IDX] = targets[task==MAIN_IDX]
            for aux in aux_task:
                idx = TASK_TO_IDX[aux]
                target_dict[idx] = targets[task==idx]

            # get the number of data for each task 
            numdata_dict[MAIN_IDX] += len(target_dict[MAIN_IDX]) 
            for aux in aux_task:
                idx = TASK_TO_IDX[aux]
                numdata_dict[idx] += len(target_dict[idx]) 
            
            # model forward
            output_dict = model(
                tokens=tokens,
                tasks=task
            ) 
            
            #######################
            # Main task
            #######################
            # calculate loss with emoji prediction task
            outputs = output_dict[MAIN_IDX]
            targets = target_dict[MAIN_IDX]
            if len(outputs) > 0:
                loss_main = loss_fn(outputs, targets) 
                loss_dict[MAIN_IDX] += loss_main.item() 

                # number of correct prediction
                _, pred_int = torch.max(outputs, dim=1) 
                acc_dict[MAIN_IDX] += torch.sum(pred_int == targets)


            #######################
            # Auxiliary task
            #######################
            for aux in aux_task:
                idx = TASK_TO_IDX[aux]
                outputs = output_dict[idx]
                targets = target_dict[idx]
                if len(outputs) > 0:
                    # calculate loss with auxiliary task
                    loss_aux = loss_fn(outputs, targets) 
                    torch_loss_dict[idx] = loss_aux
                    loss_dict[idx] += loss_aux.item()

                    # number of correct prediction
                    _, pred_int = torch.max(outputs, dim=1) 
                    acc_dict[idx] += torch.sum(pred_int == targets)

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

        # calculate accuracy and average loss
        acc_dict[MAIN_IDX] /= numdata_dict[MAIN_IDX]
        loss_dict[MAIN_IDX] /= numdata_dict[MAIN_IDX]

        for aux in aux_task: 
            idx = TASK_TO_IDX[aux]
            acc_dict[idx] /= numdata_dict[idx]
            loss_dict[idx] /= numdata_dict[idx] 

        return acc_dict, loss_dict

if __name__ == '__main__': 
    ######################################
    # configuration
    ######################################
    parser = argparse.ArgumentParser() 
    parser.add_argument('--aux_num', type=int, required=True)
    parser.add_argument('--aux_task', type=str, required=True)
    parser.add_argument('--gpu_num', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=50) 
    parser.add_argument('--save_cp', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--decay', type=bool, default=False)
    parser.add_argument('--fine_tuning', type=bool, default=False)
    parser.add_argument('--pre_trained', type=bool, default=True) 
    args = parser.parse_args()

    EPOCHS = args.num_epoch
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    patience = args.patience
    early_stop = args.early_stop
    decay = args.decay
    fine_tuning = args.fine_tuning
    pre_trained = args.pre_trained  
    save_cp = args.save_cp 
    save_path = './checkpoints/model_aux_%s.pt'%args.aux_task
    gpu_num = args.gpu_num
    AUX_NUM = args.aux_num
    AUX_TASK = args.aux_task.split(' ')
    assert len(AUX_TASK)==AUX_NUM, 'Does not match the number of auxiliary task'
    for aux in AUX_TASK:
        if aux not in ['emo', 'Ekman', 'sent']:
            raise AssertionError('Invalid format for auxiliary task. Argument should be one of "emo", "Ekman", and "sent".') 

    deepmoji_dim = 2304
    MAIN_IDX = 0
    TASK_TO_IDX = {'emo':1, 'Ekman':2, 'sent':3} 
    weight_path = './data/pytorch_model.bin'

    # set GPU device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%gpu_num
    device = torch.device('cuda') 

    ######################################
    # Load data
    ###################################### 
    print('Load dataset...')
    # Load dataset for emoji prediction
    df_train = pd.read_csv("./data/train_Twitter.csv", index_col=0, header=0)
    df_val = pd.read_csv("./data/val_Twitter.csv", index_col=0, header=0)
    df_test = pd.read_csv("./data/test_Twitter.csv", index_col=0, header=0) 

    # Load dataset for emotion detection
    df_train_ge = pd.read_csv('./data/train_GoEmotion.csv', index_col=0) 
    df_val_ge = pd.read_csv('./data/val_GoEmotion.csv', index_col=0) 
    df_test_ge = pd.read_csv('./data/test_GoEmotion.csv', index_col=0) 
    
    max_len = max([len(t.split(' ')) for t in df_train.pre_punc]) 
    st = SentenceTokenizer(max_len)

    # load dataloader
    train_data_loader = create_data_loader(AUX_NUM, AUX_TASK, df_train, df_train_ge, max_len, BATCH_SIZE, st)
    val_data_loader = create_data_loader(AUX_NUM, AUX_TASK, df_val, df_val_ge, max_len, BATCH_SIZE, st)
    test_data_loader = create_data_loader(AUX_NUM, AUX_TASK, df_test, df_test_ge, max_len, BATCH_SIZE, st)

    ######################################
    # Define model
    ######################################
    print('Define Model...')
    # load model
    torchmoji = TorchMoji()
    if pre_trained and weight_path:
        load_specific_weights(torchmoji, weight_path, exclude_names=['output_layer'])

    model = MultiEmo(AUX_TASK, torchmoji, deepmoji_dim) 
    model = model.to(device)

    # freeze the pre-trained parameters to prevent from updating parameters
    if fine_tuning:
        for param in torchmoji.parameters():
            param.requires_grad = True

    ######################################
    # Train model   
    ######################################
    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # initialize
    history = defaultdict(list)
    best_acc = 0. 

    print('Start training...')
    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        # train model with training set
        model, acc_dict_train, loss_dict_train = train_epoch(
            model,
            AUX_TASK, 
            train_data_loader,    
            loss_fn, 
            optimizer 
            )
        
        # evaluate model with validation set
        acc_dict_val, loss_dict_val = eval_model(
            model,
            AUX_TASK,
            val_data_loader,
            loss_fn 
            ) 
         
        print_result(AUX_TASK, acc_dict_train, loss_dict_train, 'Train')
        print_result(AUX_TASK, acc_dict_val, loss_dict_val, 'Val')
        
        history = save_history(history, AUX_TASK, acc_dict_train, loss_dict_train, acc_dict_val, loss_dict_val)
        
        # checkpoint for early stop
        val_acc_cp = acc_dict_val[MAIN_IDX]
        if val_acc_cp > best_acc:
            best_acc = val_acc_cp
            patience = 0
            if save_cp:
                save_checkpoint(model, save_path)
        else:
            patience += 1
            print('-' * 5, 'patience ', patience, '-'*5)
            
        
        if early_stop and patience > early_stop:
            print('early stop triggered at epoch %d!'%epoch)
            break
        
        if decay:
            optimizer = lr_decay(optimizer)

    ######################################
    # Evaluate model   
    ######################################
    # infer the test dataset
    pred_list, true_list = get_output(model, test_data_loader, device)
    
    # print the result
    k = 5
    print_metric(pred_list, true_list, 'Single score', k)
    print_metric(pred_list, true_list, 'Category score', k)
    print_metric(pred_list, true_list, 'Sentiment score', k)
