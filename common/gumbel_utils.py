import torch
import numpy as np

eps = 1e-20

# logits: NxC input vector, where N can be arbitrary and C is the number of classes, entries are unnormalized log probs
# returns: a sample from the gumbel distribution with the given temperature for the probs given in "logits"
def gumbel_softmax_sample(logits, temperature):
    u = torch.rand(logits.size())
    gumbel = - np.log(- np.log(u + eps) + eps)
    gumbel = gumbel.to(logits.device)
    y = logits + gumbel
    return torch.nn.functional.softmax( y / temperature, dim = 1)
