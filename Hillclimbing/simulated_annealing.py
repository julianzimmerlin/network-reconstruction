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

SEED = 0
SERIES_ADDRESS = r'../data/final/cml/timeseries_ba10_1k_3.5_restart.pickle'
ADJ_ADDRESS = r'../data/final/edges_ba10.pickle'
BATCH_SIZE = 100
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS = 200
NUM_GEN = 45
DETECT_EARLY_CONVERGENCE = False
FORMAT = 'standard'
USE_EVALEPOCH_FOR_GUIDED_MUTATION = True
EXPERIMENTS = 1
CONTINUATION = False
CONT_ADDRESS = '/content/drive/MyDrive/BA_Code/hillclimbing_logs/annealing/first/2021-01-27T18_56_40.854852'

torch.manual_seed(SEED)
np.random.seed(SEED)
np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)


#note: this depends on the loss function
def get_temperature(frac):
    return (1-frac) / 100

# return True if the new candidate should be accepted and False otherwise
def accept_or_reject(loss, newloss, temperature):
    if newloss < loss:
        print('smaller, so accepting')
        return True
    prob = torch.exp(-(newloss-loss) / temperature)
    print('will accept with prob ' + str(prob))
    if torch.rand(1) < prob:
        print('accepting!')
        return True
    else:
        print('rejecting.')
        return False


def calc_edge_mutation_probs_uniform(matrix):
    probs = 0.1 * torch.ones_like(matrix)
    mask = torch.eye(probs.size()[0], probs.size()[1]).byte()
    probs.masked_fill_(mask, 0)
    return probs
# -----------------------------------------------------------------------


orig_terminal = sys.stdout
exp_final_accs = list()
exp_final_tprs = list()
exp_final_fprs = list()
for _ in range(EXPERIMENTS):
    logger = lo.Logger('hillclimbing_logs/annealing/cml_prelim_final', original_terminal=orig_terminal)
    sys.stdout = logger
    print(SERIES_ADDRESS)
    print(ADJ_ADDRESS)
    print('SEED: ' + str(SEED))
    print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
    print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
    print('CONTINUATION: ' + str(CONTINUATION))
    print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
    print('USE_EVALEPOCH_FOR_GUIDED_MUTATION: ' + str(USE_EVALEPOCH_FOR_GUIDED_MUTATION))
    print('NUM_GEN: ' + str(NUM_GEN))
    print('FORMAT: ' + FORMAT)

    # initialize evaluator with given timeseries data
    evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, FORMAT, not USE_EVALEPOCH_FOR_GUIDED_MUTATION, False, DETERMINISTIC=False)
    NUM_NODES = evaluator.get_num_nodes()
    # load ground truth matrix
    with open(ADJ_ADDRESS, 'rb') as f:
        edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
        edges = torch.tensor(edges, dtype=torch.float32)
        gt_matrix = edges.to(torch.float32).to('cpu')
        np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
    tracker = tr.Tracker(gt_matrix, logger)

    start_time = time.process_time()
    if CONTINUATION:
        with open(CONT_ADDRESS+'/matrices.pickle', 'rb') as f:
            matrices = pickle.load(f)
            cand = matrices[-1]
    else:
        cand = ut.sample_undirected_matrix_uniform(NUM_NODES)

    for gen in range(NUM_GEN):
        dyn_learner = None
        optimizer = None
        loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, dyn_learner, optimizer)
        print('\n\n Tracking generation ' + str(gen))
        tracker.track(cand, loss)

        #mutation_probs = ut.calc_edge_mutation_probs_gradient(cand) if not USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.calc_edge_mutation_probs_evalepoch_nodewise(cand, dyn_learner, evaluator)
        #print(mutation_probs.detach().cpu().numpy())
        #new_cand = cand.detach().clone()
        # flip each edge with the the respective probability in mutation_probs
        #num_changes = 0
        #for i in range(mutation_probs.size()[0]):
        #    for j in range(i+1, mutation_probs.size()[1]):
        #        if np.random.random(1)[0] < mutation_probs[i,j]:
        #            num_changes += 1
        #            new_cand[i,j] = 1-new_cand[i,j]
        #            new_cand[j,i] = 1 - new_cand[j,i]
        new_cand,changed_indices = ut.exec_dynamic_step_eval(cand, dyn_learner, evaluator, loss) if USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.exec_dynamic_step_grad(cand)
        print('Candidate: ' + ut.hash_tensor(new_cand) + '. Changes:' + str(len(changed_indices)))

        new_loss, new_dyn_learner, new_optimizer = evaluator.evaluate_individual(new_cand,NUM_DYN_EPOCHS, None, None)

        T = get_temperature((gen+1)/NUM_GEN) if gen<NUM_GEN else 0

        if accept_or_reject(loss, new_loss, T):
            cand = new_cand.detach().clone().requires_grad_(True)
        else:
            cand = cand.detach().clone().requires_grad_(True)
    tracker.track(cand, loss)
    print('Time needed: ' + str(time.process_time() - start_time) + ' sec')

    exp_final_accs.append(tracker.get_last_acc())
    exp_final_fprs.append(tracker.get_last_fpr())
    exp_final_tprs.append(tracker.get_last_tpr())

print('----------------- Final results across all experiments ---------------')
print('accs: ' + str(exp_final_accs))
print('tprs: ' + str(exp_final_tprs))
print('fprs: ' + str(exp_final_fprs))
mean_acc = torch.tensor(exp_final_accs).mean().item()
mean_tpr = torch.tensor(exp_final_tprs).mean().item()
mean_fpr = torch.tensor(exp_final_fprs).mean().item()
print('mean acc: ' + str(mean_acc))
print('mean tpr: ' + str(mean_tpr))
print('mean fpr: ' + str(mean_fpr))
