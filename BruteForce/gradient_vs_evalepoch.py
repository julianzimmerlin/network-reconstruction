import evaluator as ev
import logger as lo
import copy
import pickle, time, sys
import torch
import numpy as np
import utils as ut

logger = lo.Logger('prediction_performance_logs')
sys.stdout = logger

series_address =  '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle' # '../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/SIS_FIXED/edges_ba10.pickle'
SEED = 7
NUM_DYN_EPOCHS = 200
BATCH_SIZE = 5000
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
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)

evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, True, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, USE_MAX=False)
num_nodes = evaluator.get_num_nodes()

def calc_mutation_order_gradient(matrix):
    ut.symmetrize_matrix_(matrix.grad, take_mean=True)
    minus_grad = -1 * matrix.grad
    flipped_matrix = 1 - matrix.detach()
    gradient_partially_flipped = flipped_matrix * minus_grad + matrix.detach() * matrix.grad
    tril_selection_mat = torch.ones_like(gradient_partially_flipped).tril(diagonal=0) == 1
    gradient_partially_flipped[tril_selection_mat] = -torch.ones(matrix.size()[0] * (matrix.size()[0] + 1) // 2)
    top_entries, top_indices = gradient_partially_flipped.view(-1).topk(matrix.size()[0] * (matrix.size()[0]-1) // 2)
    top_indices_rows = top_indices // matrix.size()[1]
    top_indices_cols = top_indices % matrix.size()[0]
    top_ind_stacked = torch.stack((top_indices_rows, top_indices_cols), dim=-1)
    return top_ind_stacked

def calc_mutation_order_evalepoch(matrix, dyn_learner):
    indices = list()
    losses = list()
    for i in range(matrix.size()[0]):
        for j in range(matrix.size()[1]):
            if j <= i:
                continue
            neighbor = matrix.detach().clone()
            neighbor[i,j] = 1 - neighbor[i,j]
            neighbor[j,i] = 1 - neighbor[j,i]
            _, loss = evaluator.evaluate_individual_no_training(neighbor, dyn_learner, False)
            indices.append(torch.tensor([i,j]))
            losses.append(torch.tensor(loss))
    indices_tensor = torch.stack(indices)
    losses_tensor = torch.stack(losses)
    best_losses, best_indices = losses_tensor.sort()
    indices_tensor = indices_tensor[best_indices]
    return indices_tensor

def mutate_full_matrix(matrix, p, sym=False):
    select_mat = (torch.empty((matrix.size()[0], matrix.size()[1])).uniform_() < p).to(matrix.dtype)
    flipped_mat = 1-matrix
    result = select_mat.to(matrix.dtype)*flipped_mat + (1-select_mat).to(matrix.dtype)*matrix
    if sym:
        ut.symmetrize_matrix_(result)
    return result
# ----------------------------------------------------------------

#test_matrices = [ut.sample_undirected_matrix_uniform(num_nodes) for _ in range(10)]
#test_matrices = ut.get_all_neighbors(gt_matrix)
#est_matrices = [mutate_full_matrix(gt_matrix.detach().clone(), 0.1, True).requires_grad_(True) for _ in range(10)]
matrix = gt_matrix.detach().clone()
matrix[0,6] = 1
matrix[6,0] = 1
matrix.requires_grad_(True)
test_matrices = [matrix]

for mat in test_matrices:
    dyn_learner = None
    optimizer = None
    for i in range(10):
        score, loss, dyn_learner, optimizer = evaluator.evaluate_individual(mat, NUM_DYN_EPOCHS, None, None, True)
        order_gradient = calc_mutation_order_gradient(mat)
        order_evalepoch = calc_mutation_order_evalepoch(mat, dyn_learner)
        gradient_improvements = [(gt_matrix[i,j] != mat[i,j]).item() for (i,j) in order_gradient]
        evalepoch_improvements = [(gt_matrix[i, j] != mat[i, j]).item() for (i, j) in order_evalepoch]
        print(gradient_improvements)
        print(evalepoch_improvements)
        print('are they identical? ' + str(bool((order_gradient == order_evalepoch).all().item())))
        mat = mat.detach().clone()
        mat.requires_grad_(True)
