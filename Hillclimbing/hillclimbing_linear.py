import sys
import torch
import numpy as np
import pickle
import time
import logger as lo
import evaluator as ev
import utils as ut
import tracker as tr
import copy
import search_utils as su

SEED = 5
SERIES_ADDRESS = '../data/cml/timeseries_ba10_1k.pickle'
ADJ_ADDRESS = '../data/cml/edges_ba10.pickle'
BATCH_SIZE = 500
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS = 100
DETECT_EARLY_CONVERGENCE = True
USE_OLD_DISCRETE_FORMAT = False
USE_EVALEPOCH_FOR_GUIDED_MUTATION = False
CONTINUATION = False
CONT_ADDRESS = './hill_climbing_logs/voter_ba20_100_CONT_8ep'

logger = lo.Logger('hill_climbing_logs/linear')
sys.stdout = logger
print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('CONTINUATION: ' + str(CONTINUATION))
print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
print('SEED: ' + str(SEED))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
print('USE_EVALEPOCH_FOR_GUIDED_MUTATION: ' + str(USE_EVALEPOCH_FOR_GUIDED_MUTATION))
torch.manual_seed(SEED)
np.random.seed(SEED)

# initialize evaluator with given timeseries data
evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, not USE_EVALEPOCH_FOR_GUIDED_MUTATION, True)
NUM_NODES = evaluator.get_num_nodes()
# load ground truth matrix
with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
tracker = tr.Tracker(gt_matrix, logger)

# -----------------------------------------------------------------------


start_time = time.process_time()
if CONTINUATION:
    with open(CONT_ADDRESS+'/matrices.pickle', 'rb') as f:
        matrices = pickle.load(f)
        cand = matrices[-1]
else:
    cand = ut.sample_undirected_matrix_uniform(NUM_NODES)
#DEBUG::::
cand = gt_matrix.detach().clone().requires_grad_(True)
max_changes = NUM_NODES
for gen in range(10):
    dyn_learner = None
    optimizer = None
    loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, dyn_learner, optimizer)
    tracker.track(cand, loss)

    indices = ut.calc_mutation_order_evalepoch(cand, dyn_learner, evaluator) if USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.calc_mutation_order_gradient(cand)
    indices = indices[:max_changes] # do max_changes most promising mutations for linear runtime
    print(indices)
    new_cand = cand.detach().clone()
    for index in indices:
        new_cand[list(index)] = 1 - new_cand[list(index)]
        new_cand[list(index.flip(dims=(0,)))] = 1 - new_cand[list(index.flip(dims=(0,)))]
    new_cand.requires_grad_(not USE_EVALEPOCH_FOR_GUIDED_MUTATION)

    #loss,_,_ = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, copy.deepcopy(dyn_learner), copy.deepcopy(optimizer))
    #print('Reevaluating current cand. ', end='')
    #loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, None, None)

    new_loss, new_dyn_learner, new_optimizer = evaluator.evaluate_individual(new_cand,NUM_DYN_EPOCHS, None, None)
    count_changes = 0
    cand = cand.detach().clone()
    for index in indices: # check for each mutation if it was good
        if su.compare_matrices_specific_mutation(new_cand, cand, new_loss, loss, index) < 0:
            count_changes += 1
            cand[list(index)] = new_cand.detach()[list(index)]
            cand[list(index.flip(dims=(0,)))] = new_cand.detach()[list(index.flip(dims=(0,)))]
    cand.requires_grad_(True)
    print('count_changes: ' + str(count_changes))
    if count_changes == 0:
        if max_changes > 1:
            max_changes = 1
        else:
            break
tracker.track(cand, loss)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')