import time
import torch
from tqdm import tqdm


def get_output(model, data_loader, device):
    '''
    Arguments:
      - model: the model for inference
      - data_loader: torch data loader for prediction
    Returns
      - pred_list: a list of prediction, a probability tensor for all labels
          ex. [[0.2, 0.1, 0.05, ...], [], ..., []]
      - true_list: a list of true label
    '''
    main_idx = 0
    model = model.eval()

    # initialize 
    pred_list = []
    true_list = []

    start = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # load data from test data loader
            targets = batch["label"].to(device)
            tokens = batch["token"].to(device) 
            task = batch['task'].to(device)

            # model forward
            outputs = model(
                tokens=tokens,
                tasks = task
                ) 

            # get the output and true label for main task
            targets_main = targets[task==main_idx]  
            output_main = outputs[main_idx] 

            pred_list += outputs[main_idx]
            true_list += [int(t) for t in targets_main]

    print('Cost time %f'%round(time.time() - start, 4))
    return pred_list, true_list