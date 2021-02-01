import evaluator as ev
import pickle
import torch
import numpy as np

series_address =  '../data/SIS_FIXED/timeseries_bull_5k_0.2_0.2.pickle' #'../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/SIS_FIXED/edges_bull.pickle'
dyn_learner_address = '../data/dyn_learners/SIS_bull_5k_0.2_full.pt'
SEED = 4
NUM_DYN_EPOCHS = 300
DETECT_EARLY_CONVERGENCE = True
BATCH_SIZE = 100
HIDDEN_SIZE = 128

USE_OLD_DISCRETE_FORMAT = True
print(series_address)
print(adj_address)
print('SEED: {}'.format(SEED))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('USE_OLD_DISCRETE_FORMAT: ' + str(USE_OLD_DISCRETE_FORMAT))
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
torch.manual_seed(SEED)
np.random.seed(SEED)

with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
print(gt_matrix)

evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, False, False, USE_MAX=False)
num_nodes = evaluator.get_num_nodes()

# ----------------------------------------------------------------
mat = gt_matrix.detach().clone()

loss,dyn_learner,_ = evaluator.evaluate_individual(mat, 0, None, None)
print(loss)
#torch.save(dyn_learner.state_dict(), dyn_learner_address)
torch.save(dyn_learner, dyn_learner_address)