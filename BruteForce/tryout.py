import evaluator as ev
import logger as lo
import utils as ut
import pickle, time, sys
import torch
import numpy as np

logger = lo.Logger('tryout_logs')
sys.stdout = logger

SERIES_ADDRESS = r'D:\Uni\BA\Development\data\final\cml\timeseries_bull_5k_4_onetake.pickle'
ADJ_ADDRESS = r'D:\Uni\BA\Development\data\final\netrd\Kuramoto\edges_bull.pickle'
SEED = 4
NUM_DYN_EPOCHS = 100
FORMAT='standard'
BATCH_SIZE = 100
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
mat1 = torch.zeros(num_nodes, num_nodes)
mat2 = torch.ones(num_nodes, num_nodes)
mat3 = ut.sample_undirected_matrix_uniform(num_nodes)
mat4 = mat0.detach().clone()
mat4[3,4] = 1
mat4[4,3] = 1
#mat5 = mat0.detach().clone()
#ind = np.diag_indices(mat5.shape[0])
#mat5[ind[0], ind[1]] = torch.ones(mat5.shape[0])

all_matrices = [mat0, mat4, mat1, mat2]
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
