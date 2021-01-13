import evaluator as ev
import logger as lo
import copy
import pickle, time, sys
import torch
import numpy as np

logger = lo.Logger('prediction_performance_logs')
sys.stdout = logger

series_address =  '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle' # '../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/SIS_FIXED/edges_ba10.pickle'
SEED = 1
NUM_DYN_EPOCHS = 200
BATCH_SIZE = 5000
HIDDEN_SIZE = 64
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
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)

evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, True, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, USE_MAX=False)
num_nodes = evaluator.get_num_nodes()

# ----------------------------------------------------------------

np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)
matrix = gt_matrix.detach().clone()#ut.sample_undirected_matrix_uniform(num_nodes)
matrix[0,1] = 1
matrix[1,0] = 1
matrix.requires_grad_(True)
print(matrix.detach().numpy().astype(int))
gt_matrix.requires_grad_(True)
#rand_matrix = ut.sample_undirected_matrix_uniform(num_nodes)
#print(rand_matrix.detach().numpy().astype(int))

print('Reusing dyn_learner and optimizer:')
for i in range(5):
    score3, loss3, dyn_learner3, optimizer3 = evaluator.evaluate_individual(matrix.detach().clone(),400,get_gradient=False)
    score1, loss1, dyn_learner1, optimizer1 = evaluator.evaluate_individual(gt_matrix.detach().clone(),400, dyn_learner3, optimizer3, get_gradient=False)
    print('-------------------')
print('Reusing just dyn_learner')
for i in range(5):
    score3, loss3, dyn_learner3, optimizer3 = evaluator.evaluate_individual(matrix.detach().clone(), 400,
                                                                            get_gradient=False)
    score2, loss2, dyn_learner2, optimizer2 = evaluator.evaluate_individual(gt_matrix.detach().clone(), 400,
                                                                            dyn_learner3,
                                                                            get_gradient=False)
    print('-------------------')
print('Training from scratch:')
for i in range(5):
    score2, loss2, dyn_learner2, optimizer2 = evaluator.evaluate_individual(gt_matrix.detach().clone(), 400,
                                                                            get_gradient=False)