import torch
import pickle
import numpy as np
import sys, os, datetime
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from constants import *

def load_data(series_address, batch_size = 128, split_train_test=True):
    # print('series_address: ' + series_address)

    with open(series_address,'rb') as f:
        info_train = pickle.load(f,encoding='latin1') # (#samples*2 x num_nodes) 1 or 0
    # with open(adj_address,'rb') as f:
    #    edges = pickle.load(f,encoding='latin1') # (num_nodes x num_nodes) 1 or 0

    data_x = np.zeros((int(info_train.shape[0] / 2), info_train.shape[1], 2))  # (#samples x num_nodes x 2)
    data_y = np.zeros((int(info_train.shape[0] / 2), info_train.shape[1]))  # (#samples x num_nodes)

    # Preprocessed into data formats commonly used in classification tasks
    for i in range(int(info_train.shape[0] / 2)):
        for j in range(info_train.shape[1]):
            if info_train[2 * i][j] == 0:  # put even entries in data_x
                data_x[i][j] = [1, 0]  # if it's 0, encode 1 in first feature map and 0 in second feature map
            else:
                data_x[i][j] = [0, 1]  # other way around
            if info_train[2 * i + 1][j] == 0:  # put uneven entries in data_y
                data_y[i][j] = 0  # this time no feature map shenanigans...
            else:
                data_y[i][j] = 1

    # random permutation of data
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = [data_x[i] for i in indices]
    data_y_temp = [data_y[i] for i in indices]
    data_x = np.array(data_x_temp)
    data_y = np.array(data_y_temp)

    # seperate train set and test set
    # train / val == 5 / 1
    train_len = int(data_x.shape[0] * 5 / 6) if split_train_test else data_x.shape[0]
    # seperate
    feat_train = data_x[:train_len]
    target_train = data_y[:train_len]
    feat_test = data_x[train_len:]
    target_test = data_y[train_len:]

    # change to torch.tensor
    feat_train = torch.FloatTensor(feat_train)
    feat_test = torch.FloatTensor(feat_test) # formerly Double
    target_train = torch.ByteTensor(target_train)  # formerly Long
    target_test = torch.ByteTensor(target_test)

    # put into tensor dataset
    train_data = TensorDataset(feat_train, target_train)
    test_data = TensorDataset(feat_test, target_test)

    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    return (train_data_loader, test_data_loader) if split_train_test else train_data_loader


def load_data_cont(series_address, batch_size = 128):
    # TODO this seems to be a different input format
    with open(series_address, 'rb') as f:
        data = pickle.load(f)
    print('Data shape: '+ str(data.shape))
    data = data[0:10000]
    print('Data shape: ' + str(data.shape))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader


# used for output to both console and logfile
class Logger(object):
    def __init__(self, folder_name='logs'):
        timestamp = datetime.datetime.now().isoformat()
        self.path = './' + folder_name + '/' + str(timestamp).replace(':', '_')
        os.makedirs(self.path)
        self.terminal = sys.stdout
        self.log = open(self.path + "/logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def get_path(self):
        return self.path

# calculates fpr and tpr of a matrix compared to the ground truth adjacency matrix
def calculate_tpr_fpr(ground_truth, mat, skip_diag=False):
    matrix = mat.cpu()
    tp = 0
    p = 0
    n = 0
    fp = 0
    for i in range(ground_truth.size()[0]):
        for j in range(ground_truth.size()[1]):
            if skip_diag and i == j:
                continue
            if(ground_truth[i,j] == 0):
                n += 1
                if(matrix[i,j] == 1):
                    fp += 1
            else:
                p += 1
                if(matrix[i,j] == 1):
                    tp += 1
    return (tp / p, fp / n)


# makes a matrix symmatric by copying all values from the upper half to the lower half or taking the mean values
def symmetrize_matrix_(mat, take_mean=False):
    n = mat.size()[0]
    for i in range(n):
        for j in range(i+1,n):
            if take_mean:
                val = (mat[j,i]+mat[i,j]) / 2
                mat[j,i] = mat[i,j] = val
            else:
                mat[j,i] = mat[i,j]

# uniformly samples a undirected nxn adjacency matrix
def sample_undirected_matrix_uniform(n):
    rand = torch.full((n,n),0.5,device=DEVICE_GEN).bernoulli()
    for i in range(n):
        rand[i,i] = 0
    symmetrize_matrix_(rand)
    rand.requires_grad_(True)
    return rand

def calc_guided_mutation_probs(matrix, sym=True, softmax_factor=1, allow_mut_on_diag=False):
    if sym:
        symmetrize_matrix_(matrix.grad, take_mean=True)
    minus_grad = -1 * matrix.grad
    flipped_matrix = 1 - matrix.detach()
    gradient_partially_flipped = flipped_matrix * minus_grad + matrix.detach() * matrix.grad
    #print(gradient_partially_flipped)
    softmax = torch.nn.Softmax(dim=0)
    if sym:
        triu_selection_mat = torch.ones_like(gradient_partially_flipped).triu(diagonal=int(not allow_mut_on_diag)) == 1
        triu_vec = gradient_partially_flipped[triu_selection_mat]  # get upper triangular matrix as vector
        probs_vec = softmax(triu_vec * softmax_factor)
        probs = torch.zeros_like(gradient_partially_flipped)
        probs[triu_selection_mat] = probs_vec
    else:
        if allow_mut_on_diag==False:
            print('Caution, directed guided mutation currently always allows mutation of diagonal')
        probs = softmax(gradient_partially_flipped.view(NUM_NODES ** 2) / softmax_factor).view(NUM_NODES, NUM_NODES)
    return probs

# use guided mutation. changes exactly one entry of the matrix
def mutate_matrix_guided(matrix, sym=True):
    probs = calc_guided_mutation_probs(matrix, sym=sym, softmax_factor=1 / TEMPERATURE)
    probs_vec = probs.view(-1)  # vectorize
    sample = probs_vec.multinomial(num_samples=1)
    result = matrix.detach().clone()
    i = sample // NUM_NODES
    j = sample % NUM_NODES
    result[i, j] = 1 - result[i, j]
    if sym and i != j:
        result[j,i] = 1 - result[j,i]
    result.requires_grad_(True)
    return result
