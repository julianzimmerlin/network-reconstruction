import pickle
from torch.utils.data import DataLoader
import torch
import numpy as np


# returns a single data loader (no split into training, val and test data)
def load_data(series_address, format, batch_size):
    #print('use_old_discrete_format: ' + str(use_old_discrete_format))
    with open(series_address,'rb') as f:
        data = pickle.load(f,encoding='latin1')
    # currently there are different input formats for discrete and continuous data, so we need 2 loading functions
    if format=='old':
        return load_data_old_format(data, batch_size)
    elif format=='standard':
        return load_data_standard_format(data, batch_size)
    elif format=='timeseries':
        return load_data_timeseries(data, batch_size)

# data input dimensions: (SAMPLES, NUM_NODES) numpy array
# output dimensions: (BATCH_SIZE, NUM_NODES, NUM_STEPS=2, INPUT_SIZE=1)
def load_data_timeseries(data, batch_size):
    print('num samples: ' + str(data.shape[0]))
    expanded = expand_sequence(data)
    return load_data_old_format(expanded, batch_size)

# data input dimensions: (SAMPLES, NUM_NODES, NUM_STEPS, INPUT_SIZE) torch tensor
# output dimensions: (BATCH_SIZE, NUM_NODES, NUM_STEPS, INPUT_SIZE)
def load_data_standard_format(data, batch_size):
    #data = data[0:3000]
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).to(torch.float32)
    print('num samples: ' + str(data.size()[0]))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)  # ,pin_memory=True)
    for el in data[0:100].view(data[0:100].numel()):
        if el != 0. and el != 1.:
            return data_loader, True, data.size()[1]
    return data_loader, False, data.size()[1]


# data input dimensions: (2 * SAMPLES - 2, NUM_NODES) numpy array
# output dimensions: (BATCH_SIZE, NUM_NODES, NUM_STEPS, INPUT_SIZE=2)
def load_data_old_format(data,  batch_size):
    data_split = torch.split(torch.from_numpy(data).to(torch.float32), 2)
    data_stacked = torch.stack(data_split, dim=0).permute(0,2,1)
    data_negated = 1 - data_stacked.detach().clone()
    data_onehot = torch.stack((data_negated, data_stacked), dim=-1)
    #data_onehot = data_onehot[:50]
    data_loader = DataLoader(data_onehot, batch_size=batch_size, shuffle=True)  # ,pin_memory=True)
    return data_loader, False, data.shape[1]


def expand_sequence(seq):
    return np.array([seq[0], *[val for val in seq[1:-1] for _ in (0, 1)], seq[-1]])