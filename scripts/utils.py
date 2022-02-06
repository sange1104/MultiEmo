import torch
from copy import deepcopy

MAIN_IDX = 0

def save_history(history, aux_task, acc_dict_train, loss_dict_train, acc_dict_val, loss_dict_val):
    '''save history for accuracy and loss of all tasks''' 
    
    # main task
    history['train_acc_%s'%('emoji')].append(acc_dict_train[MAIN_IDX])
    history['train_loss_%s'%('emoji')].append(loss_dict_train[MAIN_IDX])
    history['val_acc_%s'%('emoji')].append(acc_dict_val[MAIN_IDX])
    history['val_loss_%s'%('emoji')].append(loss_dict_val[MAIN_IDX]) 
    
    # auxiliary task
    for aux in aux_task:
        history['train_acc_%s'%(aux)].append(acc_dict_train[aux])
        history['train_loss_%s'%(aux)].append(loss_dict_train[aux])
        history['val_acc_%s'%(aux)].append(acc_dict_val[aux])
        history['val_loss_%s'%(aux)].append(loss_dict_val[aux]) 
    return history

def print_result(aux_task, acc_dict, loss_dict, p_type):
    '''print accuracy and loss for all tasks''' 
    aux_to_idx = {'emo':1, 'Ekman':2, 'sent':3}
    print('Task %d : %s'%(MAIN_IDX+1, 'emoji'))
    print('%s loss %.3f accuracy %.3f'%(p_type, loss_dict[MAIN_IDX], acc_dict[MAIN_IDX]))
    for i, aux in enumerate(aux_task):
        idx = aux_to_idx[aux]
        print('Task %d : %s'%(i+1, aux))
        print('%s loss %.3f accuracy %.3f'%(p_type, loss_dict[idx], acc_dict[idx]))
        print() 

def lr_decay(optimizer):
    '''decay the learning rate of the optimizer for stable training''' 
    for i in optimizer.param_groups: 
        i['lr'] *= 0.95 
    return optimizer

def save_checkpoint(model, filename):
    '''save the checkpoint during training procedure for reproduction''' 
    torch.save(deepcopy(model.state_dict()), filename)