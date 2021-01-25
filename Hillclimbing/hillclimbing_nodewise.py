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
SERIES_ADDRESS = '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle'
ADJ_ADDRESS = '../data/SIS_FIXED/edges_ba10.pickle'
BATCH_SIZE = 5000
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS = 200
NUM_TRIES = 5
NUM_NEW_CANDIDATES = 45
DETECT_EARLY_CONVERGENCE = True
RESET_DYN_LEARNER_EVERY_NTH_GEN = 1
USE_OLD_DISCRETE_FORMAT = True
USE_NODEWISE_EVALUATION = True
USE_EVALEPOCH_FOR_GUIDED_MUTATION = True
CONTINUATION = False
CONT_ADDRESS = './hill_climbing_logs/voter_ba20_100_CONT_8ep'

logger = lo.Logger('hill_climbing_logs')
sys.stdout = logger
print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('CONTINUATION: ' + str(CONTINUATION))
print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
print('SEED: ' + str(SEED))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('NUM_TRIES: ' + str(NUM_TRIES))
print('NUM_NEW_CANDIDATES: ' + str(NUM_NEW_CANDIDATES))
print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
print('RESET_DYN_LEARNER_EVERY_NTH_GEN: ' + str(RESET_DYN_LEARNER_EVERY_NTH_GEN))
print('USE_NODEWISE_EVALUATION: ' + str(USE_NODEWISE_EVALUATION))
print('USE_EVALEPOCH_FOR_GUIDED_MUTATION: ' + str(USE_EVALEPOCH_FOR_GUIDED_MUTATION))
torch.manual_seed(SEED)
np.random.seed(SEED)

# initialize evaluator with given timeseries data
evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, not USE_EVALEPOCH_FOR_GUIDED_MUTATION, USE_NODEWISE_EVALUATION)
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
found_better=True
for gen in range(100):
    if gen % RESET_DYN_LEARNER_EVERY_NTH_GEN == 0:# or not found_better:
        dyn_learner = None
        optimizer = None
        loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, dyn_learner, optimizer)
    tracker.track(cand, loss)

    indices = ut.calc_mutation_order_evalepoch(cand, dyn_learner, evaluator)
    indices = indices[:NUM_NEW_CANDIDATES]

    found_better = False
    for num_try in range(NUM_TRIES):
        if num_try == 0:
            loss,_,_ = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, copy.deepcopy(dyn_learner), copy.deepcopy(optimizer))
        else:
            print('Reevaluating current cand. ', end='')
            loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, None, None)
        for index in indices:
            new_cand = su.double_mutation(cand, dyn_learner, evaluator, first_mut=index)
            if num_try == 0:
                new_loss, new_dyn_learner, new_optimizer = evaluator.evaluate_individual(new_cand, NUM_DYN_EPOCHS,
                                                                                                    copy.deepcopy(dyn_learner),
                                                                                                    copy.deepcopy(optimizer))
            else:
                new_loss, new_dyn_learner, new_optimizer = evaluator.evaluate_individual(new_cand,NUM_DYN_EPOCHS,
                                                                                                    None, None)
            if  (su.compare_matrices_relevant_mutations(new_cand, cand, new_loss, loss) < 0 if USE_NODEWISE_EVALUATION else new_loss < loss):
                cand = new_cand
                loss = new_loss
                dyn_learner = new_dyn_learner
                optimizer = new_optimizer
                found_better = True
                break
        if found_better:
            break
        print('No better neighbor found in round ' + str(num_try))
    if not found_better:
        break

tracker.track(cand, loss)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')