import evaluator as ev
import logger as lo
import utils as ut
import pickle, time, sys
import torch
import numpy as np

logger = lo.Logger('tryout_logs')
sys.stdout = logger

series_address =  '../data/SIS_FIXED/timeseries_bull_1k_0.2_0.2.pickle' #'../data/Voter/timeseries_bull_1000.pickle'
adj_address = '../data/SIS_FIXED/edges_bull.pickle'
SEED = 3
NUM_DYN_EPOCHS = 200
DETECT_EARLY_CONVERGENCE = True
BATCH_SIZE = 1000

HIDDEN_SIZE = 128
NUM_RUNS = 3

USE_OLD_DISCRETE_FORMAT = True
print(series_address)
print(adj_address)
print('SEED: {}'.format(SEED))
print('NUM_RUNS: {}'.format(NUM_RUNS))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('USE_OLD_DISCRETE_FORMAT: ' + str(USE_OLD_DISCRETE_FORMAT))
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
torch.manual_seed(SEED)
np.random.seed(SEED)

with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)

evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, USE_MAX=False)
num_nodes = evaluator.get_num_nodes()

# ----------------------------------------------------------------

mat0 = gt_matrix.detach().clone()
mat1 = gt_matrix.detach().clone()
mat2 = gt_matrix.detach().clone()
mat1[0,4] = 1-mat1[0,4]
mat1[4,0] = 1-mat1[4,0]
mat2[0,1] = 1-mat2[0,1]
mat2[1,0] = 1-mat2[1,0]
all_matrices = [mat0, mat1, mat2]
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
        _, loss = evaluator.evaluate_individual(all_matrices[i])
        print('Time: ' + str(time.process_time() - start_time))
        sc[run] = loss
        #print(loss)
    losses_ls.append(sc)

losses = torch.stack(losses_ls)
means = losses.mean(dim=1)
print(means.numpy())
matrices_address = logger.get_path() + '/matrices.pickle'
with open(matrices_address, 'wb') as f:
    pickle.dump(all_matrices, f)

losses_address = logger.get_path() + '/losses.pickle'
with open(losses_address, 'wb') as f:
    pickle.dump(losses, f)
