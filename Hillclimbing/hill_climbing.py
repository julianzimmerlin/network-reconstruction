import sys
import torch
import numpy as np
import pickle
import time
import logger as lo
import evaluator as ev
import utils as ut
import tracker as tr
SEED = 12342
SERIES_ADDRESS = '../data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle'
ADJ_ADDRESS = '../data/SIS_FIXED/edges_ba10.pickle'
BATCH_SIZE = 5000
HIDDEN_SIZE = 128
NUM_RUNS_FAST = 1
NUM_RUNS_SLOW = 1
NUM_DYN_EPOCHS = 200
DETECT_EARLY_CONVERGENCE = True
USE_OLD_DISCRETE_FORMAT = True
USE_MAX = False
CONTINUATION = False
CONT_ADDRESS = './hill_climbing_logs/voter_ba20_100_CONT_8ep'

logger = lo.Logger('hill_climbing_logs')
sys.stdout = logger
print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('CONTINUATION: ' + str(CONTINUATION))
print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
print('SEED: ' + str(SEED))
print('NUM_RUNS_FAST: ' + str(NUM_RUNS_FAST))
print('NUM_RUNS_SLOW: ' + str(NUM_RUNS_SLOW))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
print('USE_MAX: ' + str(USE_MAX))
torch.manual_seed(SEED)
np.random.seed(SEED)

# initialize evaluator with given timeseries data
evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT=USE_OLD_DISCRETE_FORMAT, USE_MAX=USE_MAX)
NUM_NODES = evaluator.get_num_nodes()
# load ground truth matrix
with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
tracker = tr.Tracker(gt_matrix, logger)


def calc_guided_mutation_probs(matrix, sym=True, softmax_factor=1, allow_mut_on_diag=False):
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
# -----------------------------------------------------------------------


start_time = time.process_time()
if CONTINUATION:
    with open(CONT_ADDRESS+'/matrices.pickle', 'rb') as f:
        matrices = pickle.load(f)
        cand = matrices[-1]
else:
    cand = ut.sample_undirected_matrix_uniform(NUM_NODES)
for gen in range(100):
    score, loss = evaluator.eval_individual_n_times(cand, NUM_RUNS_FAST, get_gradient=True)  # maybe i should put this in the loop -> would prevent unfairness problem
    tracker.track(cand, loss, score)

    probs = calc_guided_mutation_probs(cand, softmax_factor=1)
    _, indices = probs.view(-1).topk(10)      # NUM_NODES*(NUM_NODES+1) // 2)
    found_better = False
    new_cands = list()
    new_cands_are_improvements = list()
    new_values = list()
    for index in indices:
        i = index // NUM_NODES
        j = index % NUM_NODES
        new_cand = cand.detach().clone()
        new_cand[i,j] = 1 - new_cand[i,j]
        if i != j:
            new_cand[j,i] = 1 - new_cand[j,i]
        new_cand.requires_grad_(True)
        new_cands.append(new_cand)
        new_values.append(int(new_cand[i,j].item()))
        new_cands_are_improvements.append((gt_matrix[i,j] == new_cand[i,j]).item())
    print('Which new_cands are improvements?: ' + str(new_cands_are_improvements))
    print('The new values?: ' + str(new_values))
    for index, new_cand in enumerate(new_cands):
        #print('Trying out new_cand: \n' + str(new_cand.detach().cpu().numpy().astype(int)))
        #print('Is this an improvement?: ' + str(new_cands_are_improvements[index]))
        new_score, _ = evaluator.eval_individual_n_times(new_cand, NUM_RUNS_FAST, get_gradient=False)
        #print('New score: ' + str(new_score))
        if new_score > score:
            cand = new_cand
            score = new_score
            found_better = True
            break
    if not found_better:
        print('Maybe a local minimum. Repeating with ' + str(NUM_RUNS_SLOW) + ' evaluations to make sure.')
        score,_ = evaluator.eval_individual_n_times(cand, NUM_RUNS_SLOW, get_gradient=False)
        print('Current AVG score to beat: ' + str(score))
        for new_cand in new_cands:
            new_score,_ = evaluator.eval_individual_n_times(new_cand, NUM_RUNS_SLOW, get_gradient=False)
            print('Candidate AVG score:' + str(new_score))
            if new_score > score:
                cand = new_cand
                score = new_score
                found_better = True
                break
        if not found_better:
            print('It seems to really be a local minimum')
            break


tracker.track(cand, loss, score)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')