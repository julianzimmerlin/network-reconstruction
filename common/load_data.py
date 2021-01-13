import pickle
from torch.utils.data import DataLoader
import torch


# returns a single data loader (no split into training, val and test data)
def load_data(series_address, use_old_discrete_format, batch_size):
    print('use_old_discrete_format: ' + str(use_old_discrete_format))
    with open(series_address,'rb') as f:
        data = pickle.load(f,encoding='latin1')
    # currently there are different input formats for discrete and continuous data, so we need 2 loading functions
    if use_old_discrete_format:
        return load_data_old_format(data, batch_size)
    else:
        return load_data_standard_format(data, batch_size)


# data input dimensions: (SAMPLES, NUM_NODES, NUM_STEPS, INPUT_SIZE) torch tensor
# output dimensions: (BATCH_SIZE, NUM_NODES, NUM_STEPS, INPUT_SIZE)
def load_data_standard_format(data, batch_size):
    #data = data[0:3000]
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
