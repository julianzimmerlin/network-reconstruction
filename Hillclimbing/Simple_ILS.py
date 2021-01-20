import numpy as np
import pickle, copy, time, sys, torch
import logger as lo
import evaluator as ev
import utils as ut
import tracker as tr
SEED = 21
SERIES_ADDRESS = '/content/drive/MyDrive/BA_Code/data/SIS_FIXED/timeseries_ba10_1k_0.2.pickle'
ADJ_ADDRESS = '/content/drive/MyDrive/BA_Code/data/SIS_FIXED/edges_ba10.pickle'
BATCH_SIZE = 5000
HIDDEN_SIZE = 64
NUM_DYN_EPOCHS = 200
NUM_TRIES = 3
NUM_NEW_CANDIDATES = 45
DETECT_EARLY_CONVERGENCE = True
RESET_DYN_LEARNER_EVERY_NTH_GEN = 5
USE_OLD_DISCRETE_FORMAT = True
CONTINUATION = False
CONT_ADDRESS = './hill_climbing_logs/voter_ba20_100_CONT_8ep'

logger = lo.Logger('/content/drive/MyDrive/BA_Code/ILS_logs/EXP_ILS', abs_path=True)
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
torch.manual_seed(SEED)
np.random.seed(SEED)

# initialize evaluator with given timeseries data
evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT=USE_OLD_DISCRETE_FORMAT)
NUM_NODES = evaluator.get_num_nodes()
# load ground truth matrix
with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
tracker = tr.Tracker(gt_matrix, logger)

def local_search(cand, evaluator, tracker):
    hashset = set()
    for gen in range(100):
        if gen % RESET_DYN_LEARNER_EVERY_NTH_GEN == 0:
            dyn_learner = None
            optimizer = None
            score, loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, dyn_learner,
                                                                                optimizer, False)
        tracker.track(cand, loss, score)
        hashset.add(ut.hash_tensor(cand))

        indices = ut.calc_mutation_order_evalepoch(cand, dyn_learner, evaluator)
        indices = indices[:NUM_NEW_CANDIDATES]
        new_cands = list()
        new_cands_are_improvements = list()
        new_values = list()
        for index in indices:
            i = index[0]
            j = index[1]
            new_cand = cand.detach().clone()
            new_cand[i, j] = 1 - new_cand[i, j]
            new_cand[j, i] = 1 - new_cand[j, i]
            new_cand.requires_grad_(True)
            if ut.hash_tensor(new_cand) not in hashset:
                new_cands.append(new_cand)
                new_values.append(int(new_cand[i, j].item()))
                new_cands_are_improvements.append((gt_matrix[i, j] == new_cand[i, j]).to(torch.int).item())
            else:
                print('Already visited ' + ut.hash_tensor(new_cand))
        #print('Which new_cands are improvements?: ' + str(new_cands_are_improvements))
        #print('The new values?: ' + str(new_values))
        found_better = False
        for num_try in range(NUM_TRIES):
            if num_try != 0:
                print('Reevaluating current cand. ')
                score, loss, _, _ = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, None, None, False)
            for index, new_cand in enumerate(new_cands):
                if num_try != 0:
                    new_score, new_loss, new_dyn_learner, new_optimizer = evaluator.evaluate_individual(new_cand,
                                                                                                        NUM_DYN_EPOCHS,
                                                                                                        None, None,
                                                                                                        False)
                else:
                    new_score, new_loss, new_dyn_learner, new_optimizer = evaluator.evaluate_individual(new_cand,
                                                                                                        NUM_DYN_EPOCHS,
                                                                                                        copy.deepcopy(
                                                                                                            dyn_learner),
                                                                                                        copy.deepcopy(
                                                                                                            optimizer),
                                                                                                        False)
                if new_score > score:
                    cand = new_cand
                    loss = new_loss
                    score = new_score
                    dyn_learner = new_dyn_learner
                    optimizer = new_optimizer
                    found_better = True
                    break
            if found_better:
                break
            print('No better neighbor found in round ' + str(num_try))
        if not found_better:
            break
    return cand, loss

def perturb(matrix):
    n = matrix.size()[0]
    num_flips = 5
    idx = torch.randint(low=0, high=n * (n - 1) // 2, size=(num_flips,))
    select_vector = torch.zeros(n * (n - 1) // 2)
    select_vector[idx] = 1
    select_mat = torch.zeros(n,n)
    triu_selection_mat = torch.ones_like(select_mat).triu(diagonal=1) == 1
    select_mat[triu_selection_mat] = select_vector
    flipped_mat = 1 - matrix
    result = select_mat.to(matrix.dtype) * flipped_mat + (1 - select_mat).to(matrix.dtype) * matrix
    ut.symmetrize_matrix_(result)
    return result
# -----------------------------------------------------------------------

start_time = time.process_time()
current_best = ut.sample_undirected_matrix_uniform(NUM_NODES)
current_best, current_best_loss = local_search(current_best, evaluator, tracker)
print(':::::::::::::::::::::::::::::: LOCAL SEARCH DONE :::::::::::::::::::::::::::::::::::')
for i in range(10):
    candid = perturb(current_best)
    candid_optim, candid_loss = local_search(candid, evaluator, tracker)
    print(':::::::::::::::::::::::::::::: LOCAL SEARCH DONE :::::::::::::::::::::::::::::::::::')
    if candid_loss < current_best_loss:
        print('Yay, this local optimum is the best one found so far!')
        current_best = candid_optim
        current_best_loss = candid_loss
    else:
        print('This local optimum was not better than the best one we found before.')
    print('Now best: ' + ut.hash_tensor(current_best))
tracker.track(current_best, current_best_loss)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')