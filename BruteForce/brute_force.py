import evaluator as ev
import logger as lo
import pickle, time, sys
import torch
import numpy as np
import brute_force_utils as bfu

logger = lo.Logger('brute_force_outputs/final_GPUcluster/SIS_50_50_2000_TESTSET')
sys.stdout = logger

series_address = r'..\data\final\netrd\SIS\timeseries_bull_50_0.2_dontdie.pickle'.replace('\\', '/')
adj_address = r'..\data\final\netrd\SIS\edges_bull.pickle'.replace('\\', '/')
SEED = 0
NUM_DYN_EPOCHS = 2000
BATCH_SIZE = 50
HIDDEN_SIZE = 128
NUM_RUNS = 1
FORMAT='timeseries'
USE_TESTSET=True
#USE_SHORTCUT = True
#SHORTCUT_CAP = 45
print(series_address)
print(adj_address)
print('SEED: {}'.format(SEED))
print('NUM_RUNS: {}'.format(NUM_RUNS))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('FORMAT: ' + FORMAT)
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
print('USE_TESTSET: ' + str(USE_TESTSET))
#print('USE_SHORTCUT: ' + str(USE_SHORTCUT))
#print('SHORTCUT_CAP: ' + str(SHORTCUT_CAP))
torch.manual_seed(SEED)
np.random.seed(SEED)

with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)

evaluator = ev.Evaluator(series_address, NUM_DYN_EPOCHS, False, BATCH_SIZE, HIDDEN_SIZE, FORMAT, False, False, USE_TESTSET=USE_TESTSET, USE_MAX=False)
num_nodes = evaluator.get_num_nodes()

# returns a 1024 x 5 x 5 tensor with all possible adjacency matrices
def enumerate_matrices(self_loop=False):
    matrices = list()
    for i in range(np.power(2,num_nodes*(num_nodes-1) // 2)):
        bin_i = ("{0:010b}".format(i))  # convert to binary
        mat = torch.zeros(num_nodes, num_nodes)
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

def subsample_matrices(ref_matrix, mat_list, cap):
    subsampled_mat_list = list()
    inds_by_dist = bfu.get_indices_by_dist(ref_matrix, mat_list)
    for inds in inds_by_dist:
        inds_tensor = torch.LongTensor(inds)
        inds_tensor = inds_tensor[torch.randperm(inds_tensor.size()[0])]
        for i in range(min(cap, inds_tensor.size()[0])):
            subsampled_mat_list.append(mat_list[inds_tensor[i]])
    return torch.stack(subsampled_mat_list)

# ----------------------------------------------------------------
all_matrices = enumerate_matrices()
#if USE_SHORTCUT:
#    all_matrices = subsample_matrices(gt_matrix, all_matrices, SHORTCUT_CAP)

matrices_address = logger.get_path() + '/matrices.pickle'
with open(matrices_address, 'wb') as f:
    pickle.dump(all_matrices, f)

losses_ls = list()
for i in range(all_matrices.size()[0]):
    sc = torch.zeros(NUM_RUNS)
    for run in range(NUM_RUNS):
        print('evaluating: matrix ' + str(i) + ', run ' + str(run) + '. ', end='')
        #start_time= time.process_time()
        loss,_,_ = evaluator.evaluate_individual(all_matrices[i])
        #print('Time: ' + str(time.process_time() - start_time))
        sc[run] = loss
        #print(loss)
    losses_ls.append(sc)

    losses = torch.stack(losses_ls)
    losses_address = logger.get_path() + '/losses.pickle'
    with open(losses_address, 'wb') as f:
        pickle.dump(losses, f)






