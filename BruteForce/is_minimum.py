import torch
import pickle
import brute_force_utils as bfu
import evaluator as ev
import numpy as np
import utils as ut
import logger as lo
import sys, os

series_address = '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle'
matrices_address = r'D:\Uni\BA\ColabOutputs\hillclimbing_debug\matrices.pickle'
gt_address = '../data/SIS_FIXED/edges_ba10.pickle'
SEED = 2
NUM_DYN_EPOCHS = 200
DETECT_EARLY_CONVERGENCE = True
BATCH_SIZE = 5000
HIDDEN_SIZE = 64
NUM_RUNS = 10
USE_OLD_DISCRETE_FORMAT = True

logger = lo.Logger('is_minimum_logs')
sys.stdout = logger
print(series_address)
print(matrices_address)
print('SEED: {}'.format(SEED))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
print('USE_OLD_DISCRETE_FORMAT: ' + str(USE_OLD_DISCRETE_FORMAT))
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
torch.manual_seed(SEED)
np.random.seed(SEED)

with open(matrices_address, 'rb') as f:
    matrices = pickle.load(f, encoding='latin1')
with open(gt_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')


evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, USE_MAX=False)
NUM_NODES = evaluator.get_num_nodes()

def calc_guided_mutation_probs(matrix, sym=True, softmax_factor=1., allow_mut_on_diag=False):
    if sym:
        ut.symmetrize_matrix_(matrix.grad, take_mean=True)
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

# --------------------------------

matrices = [matrices[-1]]
np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)
print('matrices length: ' + str(len(matrices)))
unique_matrices = list()
for mat in matrices:
    if not any([torch.eq(mat, umat).all() for umat in unique_matrices]):
        unique_matrices.append(mat.detach().clone().requires_grad_(True))
print('unique_matrices_length: ' + str(len(unique_matrices)))
with open(logger.get_path() + '/unique_matrices.pickle', 'wb') as f:
    pickle.dump(unique_matrices, f)

for umat_num, umat in enumerate(unique_matrices):
    neighbors = ut.get_all_neighbors(umat)
    all_matrices = [umat] + neighbors
    losses = torch.zeros(len(all_matrices), NUM_RUNS)
    for i, mat in enumerate(all_matrices):
        for run in range(NUM_RUNS):
            print('matrix ' + str(i) + ', run ' + str(run) + ': ', end='')
            _,loss,_,_ = evaluator.evaluate_individual(mat, get_gradient=True)
            losses[i,run] = loss

    means = losses.mean(dim=1)
    cand_is_minimum = (means[1:] >= means[0]).all()
    print('Candidate:')
    print(umat.detach().cpu().numpy().astype(int))
    acc = ut.calculate_accuracy(gt_matrix, umat)
    print('ACC: ' + str(acc))
    print('Is this a minimum? ' + str(bool(cand_is_minimum.item())))
    neighbor_indices_not_in_u_m = [i for (i, mat) in enumerate(all_matrices) if not any([torch.eq(mat, umat).all() for umat in unique_matrices])]
    cand_is_minimum_wo_unique_mat = (means[neighbor_indices_not_in_u_m] >= means[0]).all()
    print('Is this a minimum without neighbors in unique_matrices? ' + str(bool(cand_is_minimum_wo_unique_mat)))
    print('Here are the neighbors which have a lower mean loss:')
    for i, lo in enumerate(means):
        if lo < means[0]:
            better_matrix = all_matrices[i]
            print(better_matrix.detach().cpu().numpy().astype(int))
            is_in_unique_matrices = any([torch.eq(better_matrix, umat).all() for umat in unique_matrices])
            print('Is this one in unique_matrices? ' + str(bool(is_in_unique_matrices)))
    print('Here are the mutation probs: ')
    probs = calc_guided_mutation_probs(umat, sym=True, softmax_factor=25000)
    print(probs.detach().cpu().numpy())
    print('---------------------------------------------------------')

    os.makedirs(logger.get_path() + '/umat' + str(umat_num))
    with open(logger.get_path() + '/umat' + str(umat_num) + '/losses.pickle', 'wb') as f:
        pickle.dump(losses, f)
    with open(logger.get_path() + '/umat' + str(umat_num) + '/means.pickle', 'wb') as f:
        pickle.dump(means, f)
    with open(logger.get_path() + '/umat' + str(umat_num) + '/all_matrices.pickle', 'wb') as f:
        pickle.dump(all_matrices, f)