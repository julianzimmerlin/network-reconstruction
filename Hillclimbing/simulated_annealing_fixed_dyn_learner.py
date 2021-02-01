import sys
import torch
import numpy as np
import pickle
import time
import logger as lo
import evaluator as ev
import utils as ut
import tracker as tr
import models as mo

SEED = 8
SERIES_ADDRESS = '../data/cml/timeseries_ba20_1k.pickle'
ADJ_ADDRESS = '../data/cml/edges_ba20.pickle'
DYN_LEARNER_ADDRESS = r'D:\Uni\BA\Development\data\dyn_learners\cml_ba10_50k.pt'
BATCH_SIZE = 5000
HIDDEN_SIZE = 128
NUM_GEN = 30
DETECT_EARLY_CONVERGENCE = True
USE_OLD_DISCRETE_FORMAT = False
USE_EVALEPOCH_FOR_GUIDED_MUTATION = False
CONTINUATION = False
CONT_ADDRESS = '/content/drive/MyDrive/BA_Code/hillclimbing_logs/annealing/first/2021-01-27T18_56_40.854852'

logger = lo.Logger('hill_climbing_logs/fixed_dyn_learner')
sys.stdout = logger
print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('DYN_LEARNER: ' + DYN_LEARNER_ADDRESS)
print('SEED: ' + str(SEED))
print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
print('CONTINUATION: ' + str(CONTINUATION))
print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
print('USE_EVALEPOCH_FOR_GUIDED_MUTATION: ' + str(USE_EVALEPOCH_FOR_GUIDED_MUTATION))
print('NUM_GEN: ' + str(NUM_GEN))
torch.manual_seed(SEED)
np.random.seed(SEED)
np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)

# initialize evaluator with given timeseries data
evaluator = ev.Evaluator(SERIES_ADDRESS, 0, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, not USE_EVALEPOCH_FOR_GUIDED_MUTATION, False)
NUM_NODES = evaluator.get_num_nodes()
# load ground truth matrix
with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
tracker = tr.Tracker(gt_matrix, logger)


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
# -----------------------------------------------------------------------


start_time = time.process_time()
if CONTINUATION:
    with open(CONT_ADDRESS+'/matrices.pickle', 'rb') as f:
        matrices = pickle.load(f)
        cand = matrices[-1]
else:
    cand = ut.sample_undirected_matrix_uniform(NUM_NODES)

# load dyn_learner
#dyn_learner = mo.GraphNetwork(evaluator.get_input_size(), HIDDEN_SIZE, evaluator.is_continuous())
#dyn_learner.load_state_dict(torch.load(DYN_LEARNER_ADDRESS))
dyn_learner = torch.load(DYN_LEARNER_ADDRESS)
dyn_learner.eval()
dyn_learner = dyn_learner.cuda()

for gen in range(NUM_GEN+100):
    loss = evaluator.evaluate_individual_no_training(cand, dyn_learner)
    print('\n\n Tracking generation ' + str(gen))
    tracker.track(cand, loss)

    mutation_probs = ut.calc_edge_mutation_probs_gradient(cand) if not USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.calc_edge_mutation_probs_evalepoch_nodewise(cand, dyn_learner, evaluator)
    print(mutation_probs.detach().cpu().numpy())
    new_cand = cand.detach().clone()
    # flip each edge with the the respective probability in mutation_probs
    num_changes = 0
    for i in range(mutation_probs.size()[0]):
        for j in range(i+1, mutation_probs.size()[1]):
            if np.random.random(1)[0] < mutation_probs[i,j]:
                num_changes += 1
                new_cand[i,j] = 1-new_cand[i,j]
                new_cand[j,i] = 1 - new_cand[j,i]
    print('Candidate: ' + ut.hash_tensor(new_cand) + '. Changes:' + str(num_changes))

    new_loss = evaluator.evaluate_individual_no_training(new_cand, dyn_learner)

    T = get_temperature((gen+1)/NUM_GEN) if gen<NUM_GEN else 0

    if accept_or_reject(loss, new_loss, T):
        cand = new_cand.detach().clone().requires_grad_(True)
    else:
        cand = cand.detach().clone().requires_grad_(True)
tracker.track(cand, loss)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')