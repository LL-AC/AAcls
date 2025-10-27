
import torch.nn as nn
import torch

def build_criterion(args):
    
    criterion = args.ctiterion.lower()
    
    if criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'mse':
        criterion = nn.MSELoss()
    elif criterion == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif criterion == 'in':
        # loss calculate by model
        return None
    else:
        raise NotImplementedError(f"criterion {criterion} is not implemented")
    
    return criterion