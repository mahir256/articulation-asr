import numpy as np
import torch.nn as nn

class LinearND(nn.Module):
    r"""
    """

    def __init__(self, *args):
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)
    
    def forward(self, x):
        r"""
        """
        size = x.size()
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

