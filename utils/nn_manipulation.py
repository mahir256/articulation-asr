import torch.nn as nn

def graves_const(layer):
    r"""Initializes weights per the recommendation in https://doi.org/10.1145/1143844.1143891,
    section 5.2 "Experimental Setup".
    """
    if(type(layer) == nn.Linear):
        nn.init.uniform_(layer.weight,-0.1,0.1)
        nn.init.uniform_(layer.bias,-0.1,0.1)
    elif(type(layer) == nn.GRU):
        for name, param in layer.named_parameters(): 
            if 'weight' in name or 'bias' in name:
                nn.init.uniform_(param,-0.1,0.1);
    elif(type(layer) == nn.Conv2d):
        nn.init.uniform_(layer.weight,0,0.1);
        nn.init.uniform_(layer.bias,0,0.1);

# TODO: make annealing rate customizable by epoch, rate, and other things
# TODO: pick schedulers from torch.optim.lr_scheduler and use them if possible
def learning_rate_adjustment(optimizer):
    r"""Reduces the learning rate of each parameter in a given optimizer
    by a factor of 1.1.
    """
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / 1.1
