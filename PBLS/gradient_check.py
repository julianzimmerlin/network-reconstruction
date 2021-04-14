import sys, pickle
import torch
import evaluator as ev
import logger as lo
import numpy as np
import tracker as tr
import utils as ut

SEED = 1234
SERIES_ADDRESS = '../data/SIS/timeseries_bull_5k_0.5_0.5.pickle'
ADJ_ADDRESS = '../data/SIS/edges_bull.pickle'
BATCH_SIZE = 5000
HIDDEN_SIZE = 128
#NUM_RUNS = 1
NUM_DYN_EPOCHS = 30
TEMPERATURE = 1 / 3000  # is independent of amount of training data because gradients are normalized
USE_OLD_DISCRETE_FORMAT = True

print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('SEED: ' + str(SEED))
#print('NUM_RUNS: ' + str(NUM_RUNS))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('TEMPERATURE: ' + str(TEMPERATURE))

logger = lo.Logger('gradient_check_logs')
sys.stdout = logger
torch.manual_seed(SEED)
np.random.seed(SEED)

evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT)
NUM_SAMPLES = evaluator.get_num_samples()
NUM_BATCHES = evaluator.get_num_batches()
NUM_NODES = evaluator.get_num_nodes()
# load ground truth matrix
with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')


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

# -----------------------------------------------------------
np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)
#matrix = gt_matrix
#matrix[1,2] = 1-matrix[1,2]
#matrix[2,1] = 1-matrix[2,1]
matrix = ut.sample_undirected_matrix_uniform(NUM_NODES)
print(matrix.detach().cpu().numpy().astype(int))
matrix.requires_grad_(True)

evaluator.evaluate_individual(matrix)
print(calc_guided_mutation_probs(matrix, softmax_factor=1 / TEMPERATURE).data.cpu().numpy())

for i in range(5):
    matrix.grad.zero_()
    evaluator.evaluate_individual(matrix)
    print(calc_guided_mutation_probs(matrix, softmax_factor=1 / TEMPERATURE).data.cpu().numpy())


