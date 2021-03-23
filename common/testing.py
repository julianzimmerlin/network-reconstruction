import evaluator as ev
import pickle
import torch
import numpy as np
import search_utils as su

series_address =  '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle' #'../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/SIS_FIXED/edges_ba10.pickle'
SEED = 3
NUM_DYN_EPOCHS = 200
DETECT_EARLY_CONVERGENCE = True
BATCH_SIZE = 1000

HIDDEN_SIZE = 128
NUM_RUNS = 3

USE_OLD_DISCRETE_FORMAT = True
print(series_address)
print(adj_address)
print('SEED: {}'.format(SEED))
print('NUM_RUNS: {}'.format(NUM_RUNS))
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

#evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, USE_MAX=False)
#num_nodes = evaluator.get_num_nodes()

# ----------------------------------------------------------------
#mat = gt_matrix.detach().clone()
#mat[0,1] = 1
#mat[1,0] = 1

#gt_loss,_,_ = evaluator.evaluate_individual(gt_matrix, 0, None, None, False, True)
#loss,_,_ = evaluator.evaluate_individual(mat, 0, None, None, False, True)
#print(gt_loss)
#print(loss)
#res = su.compare_matrices_relevant_mutations(gt_matrix, mat, gt_loss, loss)
#print(res)

filepath = r"D:\Uni\BA\Development\GA\GA_logs\final\cml_ba20_1k_4_restart_shortcut_eval\2021-03-11T14_43_20.718510"
with open(filepath+'/all_populations.pickle') as f:
    matrices = pickle.load(f)
print(matrices)