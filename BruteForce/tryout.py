import evaluator as ev
import logger as lo
import utils as ut
import pickle, time, sys
import torch
import numpy as np

logger = lo.Logger('tryout_logs')
sys.stdout = logger

SERIES_ADDRESS = '../data/Voter/new/timeseries_ba20_100.pickle'
ADJ_ADDRESS = '../data/Voter/new/edges_ba20.pickle'
SEED = 3
NUM_DYN_EPOCHS = 3000
FORMAT='old'
BATCH_SIZE = 5000
DETECT_EARLY_CONVERGENCE = False

HIDDEN_SIZE = 128
NUM_RUNS = 1

USE_OLD_DISCRETE_FORMAT = True
print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('SEED: {}'.format(SEED))
print('NUM_RUNS: {}'.format(NUM_RUNS))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('FORMAT: ' + str(FORMAT))
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
torch.manual_seed(SEED)
np.random.seed(SEED)

with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)

evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, FORMAT, False, False, USE_MAX=False)
num_nodes = evaluator.get_num_nodes()

# ----------------------------------------------------------------
#addr = r'D:\Uni\BA\Development\SigmoidGraphNetwork\GGN_logs\netrd\SIS_ba10\2021-02-11T12_33_42.903541\matrices.pickle'
#with open(addr, 'rb') as f:
#    matrices = pickle.load(f)
#mat1 = matrices[-1]
mat0 = gt_matrix.detach().clone()
all_matrices = [mat0]
all_matrices.append(ut.sample_undirected_matrix_uniform(num_nodes))
for m in all_matrices:
    m.requires_grad_(True)
    print(m.detach().cpu().numpy().astype(int))

losses_ls = list()
for i in range(len(all_matrices)):
    sc = torch.zeros(NUM_RUNS)
    for run in range(NUM_RUNS):
        print('evaluating: matrix ' + str(i) + ', run ' + str(run) + '. ', end='')
        start_time= time.process_time()
        loss,_,_ = evaluator.evaluate_individual(all_matrices[i])
        print('Time: ' + str(time.process_time() - start_time))
        sc[run] = loss
        print(loss)
    losses_ls.append(sc)

losses = torch.stack(losses_ls)
means = losses.mean(dim=1)
print(means.numpy())
#matrices_address = logger.get_path() + '/matrices.pickle'
#with open(matrices_address, 'wb') as f:
#    pickle.dump(all_matrices, f)

#losses_address = logger.get_path() + '/losses.pickle'
#with open(losses_address, 'wb') as f:
#    pickle.dump(losses, f)
