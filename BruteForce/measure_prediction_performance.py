import evaluator as ev
import logger as lo
import utils as ut
import pickle, time, sys
import torch
import numpy as np

logger = lo.Logger('prediction_performance_logs')
sys.stdout = logger

series_address =  '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle' # '../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/SIS_FIXED/edges_ba10.pickle'
SEED = 5
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
rand_matrix = ut.sample_undirected_matrix_uniform(num_nodes)
print(rand_matrix.detach().numpy().astype(int))

_,_,dyn_learner0 = evaluator.evaluate_individual(matrix)
for i in range(5):
    _,loss_0,_ = evaluator.evaluate_individual_no_training(matrix, dyn_learner0, False)
    _,loss_1,_ = evaluator.evaluate_individual_no_training(gt_matrix, dyn_learner0, False)
    _,loss_2, _ = evaluator.evaluate_individual_no_training(rand_matrix, dyn_learner0, False)
    print('.')
print('------------')
_,_,dyn_learner0 = evaluator.evaluate_individual(gt_matrix.requires_grad_(True))
for i in range(5):
    _,loss_0,_ = evaluator.evaluate_individual_no_training(matrix, dyn_learner0, False)
    _,loss_1,_ = evaluator.evaluate_individual_no_training(gt_matrix, dyn_learner0, False)
    _, loss_2, _ = evaluator.evaluate_individual_no_training(rand_matrix, dyn_learner0, False)
    print('.')
print('------------')
_,_,dyn_learner0 = evaluator.evaluate_individual(rand_matrix)
for i in range(5):
    _,loss_0,_ = evaluator.evaluate_individual_no_training(matrix, dyn_learner0, False)
    _,loss_1,_ = evaluator.evaluate_individual_no_training(gt_matrix, dyn_learner0, False)
    _, loss_2, _ = evaluator.evaluate_individual_no_training(rand_matrix, dyn_learner0, False)
    print('.')

# input shape: (BATCH_SIZE, NUM_NODES, INPUT_SIZE). INPUT_SIZE is 2 for binary dynamics.
# matrix shape: (NUM_NODES, NUM_NODES, BATCH_SIZE, 1 , 1)
#mat = matrix.repeat(1,1,1)
#input = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32) # here, 1 stands for infected and 0 for non-infected
#input = torch.stack([1-input,input], dim=-1)
#input = torch.unsqueeze(input, 0)
#log_output = dyn_learner(input.cuda(), mat.cuda())
#probs = torch.exp(log_output)
#print('State 0 Probability, State 1 probability')
#print(probs.detach().cpu().numpy())