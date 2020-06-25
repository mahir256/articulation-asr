import torch.nn as nn

def graves_const(layer):
    r"""Initializes weights per the recommendation in Graves et al. (TODO link to paper)
    """
    try:
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
    except AttributeError:
        print("Didn't work for", type(layer))
        pass

# TODO: make annealing rate customizable by epoch, rate, and other things
def learning_rate_adjustment(optimizer):
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / 1.1
