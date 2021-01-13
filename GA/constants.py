import torch
import numpy as np

NUM_NODES = 20
CONTINUOUS = False
USE_GPU_GEN = False
DEVICE_GEN = 'cuda' if USE_GPU_GEN else 'cpu'
POP_SIZE = 10
NEWPOP_SIZE = 30
NUM_GEN = 100
MUT_PROB = 1 / (NUM_NODES * (NUM_NODES+1) / 2)  # expected value is 1 mutation in the matrix
CROSS_PROB = 0.
CONTINUATION = True
CONT_PATH = './logs/voter_grid_60'
TEMPERATURE = 1 / 50

USE_GPU_DYN = True
# DEVICE_DYN = 'cuda' if USE_GPU_DYN else 'cpu'
NUM_DYN_EPOCHS = 3
BATCH_SIZE = 20
HIDDEN_SIZE = 64 # not working

SEED = 3
torch.manual_seed(SEED)
np.random.seed(SEED)
