import evaluator as ev
import logger as lo
import utils as ut
import pickle, time, sys
import torch
import numpy as np

logger = lo.Logger('prediction_performance_logs')
sys.stdout = logger

series_address =  r'..\data\final\netrd\SIS\timeseries_bull_5k_0.2_dontdie.pickle' # '../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/final/edges_bull.pickle'
SEED = 3
NUM_DYN_EPOCHS = 40
BATCH_SIZE = 100
HIDDEN_SIZE = 128
FORMAT='timeseries'
print(series_address)
print(adj_address)
print('SEED: {}'.format(SEED))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('FORMAT: ' + str(FORMAT))
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
torch.manual_seed(SEED)
np.random.seed(SEED)

with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)

evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, True, BATCH_SIZE, HIDDEN_SIZE, FORMAT, True, False, DETERMINISTIC=True)
num_nodes = evaluator.get_num_nodes()

# ----------------------------------------------------------------

np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)
matrix = gt_matrix.detach().clone()
matrix[0,4] = 1-matrix[0,4]
matrix[4,0] = 1-matrix[4,0]
matrix[1,3] = 1- matrix[1,3]
matrix[3,1] = 1-matrix[3,1]
matrix[1,4] = 1- matrix[1,4]
matrix[4,1] = 1-matrix[4,1]
matrix[3,4] = 1- matrix[3,4]
matrix[4,3] = 1-matrix[4,3]

matrix.requires_grad_(True)
print(matrix.detach().numpy().astype(int))
#rand_matrix = ut.sample_undirected_matrix_uniform(num_nodes)
#print(rand_matrix.detach().numpy().astype(int))

loss,dyn_learner0,_ = evaluator.evaluate_individual(matrix)
print(loss)


# Sgrad = ut.calc_S_grad(matrix)
# print(Sgrad.detach().numpy())
# dynamic_probs_grad = ut.dynamic_step_probabilities_grad(matrix)
# print(dynamic_probs_grad.detach().numpy())
# single_probs_grad = ut.single_step_probabilities_grad(matrix)
# print(single_probs_grad.detach().numpy())
dynamic_step_grad, ind2 = ut.exec_dynamic_step_grad(matrix)
print(dynamic_step_grad.detach().numpy().astype(int))
print(ind2)

loss,dyn_learner0,_ = evaluator.evaluate_individual(matrix)
print(loss)
dynamic_step_grad, ind2 = ut.exec_dynamic_step_grad(matrix)
print(dynamic_step_grad.detach().numpy().astype(int))
print(ind2)
# single_step_grad, ind3 = ut.exec_single_step_grad(matrix)
# print(single_step_grad.detach().numpy().astype(int))
# print(ind3)
#
# print('------------------------------------------')

# Seval = ut.calc_S_eval(matrix, dyn_learner0, evaluator, loss)
# print(Seval.detach().numpy())
# dynamic_probs_eval = ut.dynamic_step_probabilities_eval(matrix, dyn_learner0, evaluator, loss)
# print(dynamic_probs_eval.detach().numpy())
# single_probs_eval = ut.single_step_probabilities_eval(matrix, dyn_learner0, evaluator, loss)
# #print(single_probs_eval.detach().numpy())
# dynamic_step_eval, ind0 = ut.exec_dynamic_step_eval(matrix, dyn_learner0, evaluator, loss)
# print(dynamic_step_eval.detach().numpy().astype(int))
# print(ind0)
# single_step_eval, ind1 = ut.exec_single_step_eval(matrix, dyn_learner0, evaluator, loss)
# print(single_step_eval.detach().numpy().astype(int))
# print(ind1)
#
# print('------------------------------------------')
# single_step_random, ind4 = ut.exec_single_step_random(matrix)
# print(single_step_random.detach().numpy().astype(int))
# print(ind4)


#for i in range(5):
#    _,loss_0,_ = evaluator.evaluate_individual_no_training(matrix, dyn_learner0, False)
#    _,loss_1,_ = evaluator.evaluate_individual_no_training(gt_matrix, dyn_learner0, False)
#    _,loss_2, _ = evaluator.evaluate_individual_no_training(rand_matrix, dyn_learner0, False)
#    print('.')
#print('------------')


# input shape: (BATCH_SIZE, NUM_NODES, INPUT_SIZE). INPUT_SIZE is 2 for binary dynamics.
# matrix shape: (NUM_NODES, NUM_NODES, BATCH_SIZE, 1 , 1)
#mat = matrix.repeat(1,1,1)
#input = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32) # here, 1 stands for infected and 0 for non-infected
#input = torch.stack([1-input,input], dim=-1)
#input = torch.unsqueeze(input, 0)
#log_output = dyn_learner0(input.cuda(), mat.cuda())
#probs = torch.exp(log_output)
#print('State 0 Probability, State 1 probability')
#print(probs.detach().cpu().numpy())