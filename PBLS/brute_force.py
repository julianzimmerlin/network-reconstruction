import evaluator as ev, tools
import pickle, sys
from constants import *

logger = tools.Logger('brute_force_logs')
sys.stdout = logger

# initialize evaluator with given timeseries
series_address = './data/VOTER/timeseries_bull_3000.pickle'
evaluator = ev.Evaluator(series_address=series_address)

# load ground truth mat
adj_address = './data/VOTER/edges_bull.pickle'
with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')

print(gt_matrix)

# returns a 1024 x 5 x 5 tensor with all possible adjacency matrices
def enumerate_matrices(self_loop=False):
    matrices = list()
    for i in range(np.power(2,NUM_NODES*(NUM_NODES-1) // 2)):
        bin_i = ("{0:010b}".format(i))  # convert to binary
        mat = torch.zeros(NUM_NODES, NUM_NODES)
        mat[0, 1] = mat[1, 0] = int(bin_i[0])
        mat[0, 2] = mat[2, 0] = int(bin_i[1])
        mat[0, 3] = mat[3, 0] = int(bin_i[2])
        mat[0, 4] = mat[4, 0] = int(bin_i[3])
        mat[1, 2] = mat[2, 1] = int(bin_i[4])
        mat[1, 3] = mat[3, 1] = int(bin_i[5])
        mat[1, 4] = mat[4, 1] = int(bin_i[6])
        mat[2, 3] = mat[3, 2] = int(bin_i[7])
        mat[2, 4] = mat[4, 2] = int(bin_i[8])
        mat[3, 4] = mat[4, 3] = int(bin_i[9])
        if self_loop:
            mat[0,0] = mat[1,1] = mat[2,2] = mat[3,3] = mat[4,4] = 1
        matrices.append(mat)
    return torch.stack(matrices)

def enumerate_matrices_check_selfloops(mat):
    matrices = list()
    for i in range(np.power(2,NUM_NODES)):
        bin_i = ("{0:05b}".format(i))  # convert to binary
        newmat = mat.detach().clone()
        for j in range(NUM_NODES):
            newmat[j,j] = int(bin_i[j])
        matrices.append(newmat)
    return torch.stack(matrices)

# ----------------------------------------------------------------
num_runs = 10
use_loss_for_eval = True

print('SEED: {}'.format(SEED))
print('num_runs: {}'.format(num_runs))
print('use_loss_for_eval: {}'.format(use_loss_for_eval))

all_matrices = enumerate_matrices_check_selfloops(gt_matrix)
scores_ls = list()
for i in range(len(all_matrices)):
    sc = torch.zeros(num_runs)
    for run in range(num_runs):
        print('evaluating: matrix ' + str(i) + ', run ' + str(run) + '. ', end='')
        sc[run] = evaluator.evaluate_individual_by_loss(all_matrices[i]) if use_loss_for_eval \
            else evaluator.evaluate_individual_testset(all_matrices[i])
    scores_ls.append(sc)

scores = torch.stack(scores_ls)

matrices_address = logger.get_path() + '/matrices.pickle'
with open(matrices_address, 'wb') as f:
    pickle.dump(all_matrices, f)

scores_address = logger.get_path() + '/scores.pickle'
with open(scores_address, 'wb') as f:
    pickle.dump(scores, f)
