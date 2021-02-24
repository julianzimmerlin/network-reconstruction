import pickle, sys
import logger as lo
import torch
import numpy as np
import brute_force_utils as bfu

logger = lo.Logger('analysis_outputs/final/CML_5k_4_1take_20')
sys.stdout = logger

#data_address = './brute_force_outputs/SIS_bull_5k_0.5_0.5'
data_address = r'D:\Uni\BA\ColabOutputs\final\bruteforce\CML_5k_4_1take_20\merged'
adj_address = '../data/final/netrd/SIS/edges_bull.pickle'
GET_MINIMA = True
print(data_address)
print(adj_address)
print('GET_MINIMA: ' + str(GET_MINIMA))

with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
print(gt_matrix)


# ----------------------------------------------

matrices_address = data_address + '/matrices.pickle'
with open(matrices_address, 'rb') as f:
    matrices = pickle.load(f)
losses_address = data_address + '/losses.pickle'
with open(losses_address, 'rb') as f:
    scores = pickle.load(f)

# summary statistics per matrix
mean_per_matrix = scores.mean(dim=1)
std_per_matrix = scores.std(dim=1)
max_per_matrix,_ = scores.max(dim=1)
min_per_matrix,_ = scores.min(dim=1)
span_per_matrix = max_per_matrix-min_per_matrix

# statistics across all matrices
mean_all = mean_per_matrix.mean()
mean_span_all = span_per_matrix.mean()
max_span_all = span_per_matrix.max()
mean_std_all = std_per_matrix.mean()
# best matrices
(best_matrices_scores, best_indices) = mean_per_matrix.topk(20, largest = not GET_MINIMA)
num_changes_for_topk = torch.zeros(len(best_indices))
for i in range(len(best_indices)):
    num_changes_for_topk[i] = bfu.calc_num_flips(gt_matrix, matrices[best_indices[i]])

# compare by hamming distance to gt_matrix
inds_by_dist = bfu.get_indices_by_dist(gt_matrix, matrices)
num_categ = len(inds_by_dist)

mean_by_dist = torch.zeros(num_categ)
max_by_dist = torch.zeros(num_categ)
min_by_dist = torch.zeros(num_categ)
mean_span_by_dist = torch.zeros(num_categ)
mean_std_by_dist = torch.zeros(num_categ)
std_mean_by_dist = torch.zeros(num_categ)
max_mean_by_dist = torch.zeros(num_categ)
min_mean_by_dist = torch.zeros(num_categ)

for i in range(num_categ):
    print(i)
    inds = inds_by_dist[i]
    print(inds)
    mean_by_dist[i] = mean_per_matrix[inds].mean()
    max_by_dist[i] = max_per_matrix[inds].max()
    min_by_dist[i] = min_per_matrix[inds].min()
    mean_span_by_dist[i] = span_per_matrix[inds].mean()
    mean_std_by_dist[i] = std_per_matrix[inds].mean()
    std_mean_by_dist[i] = mean_per_matrix[inds].std()
    max_mean_by_dist[i] = mean_per_matrix[inds].max()
    min_mean_by_dist[i] = mean_per_matrix[inds].min()

# print and save results

np.set_printoptions(precision=10, suppress=True, linewidth=200)
print('--------------------------------- RESULTS ----------------------------------')
print('Mean score: ' + str(mean_all.numpy()))
print('Mean span across runs: ' + str(mean_span_all.numpy()))
print('Max span across runs: ' + str(max_span_all.numpy()))
print('Mean Standard Dev across runs: ' + str(mean_std_all.numpy()))
print('------------------')
print('Top 20: ')
best_table = torch.stack([best_indices.float(), best_matrices_scores, num_changes_for_topk], dim=1)
print('Index    Mean(Score) NumOfChanges')
print(best_table.numpy())
for i in range(20):
    print(str(i) + 'th best matrix:')
    print(matrices[best_indices[i]].numpy())
print('------------------')
# table by Num_changes
Num_changes = torch.tensor(list(range(num_categ)), dtype=torch.float32)
out_table = torch.stack([Num_changes, mean_by_dist, max_by_dist, min_by_dist, mean_span_by_dist, mean_std_by_dist,\
                         std_mean_by_dist, max_mean_by_dist, min_mean_by_dist], dim=1)
print('Results by Number of changes (hamming distance) from ground truth matrix')
print('NumOfChanges Mean(Scores) Max(Scores) Min(Scores) Mean(Span(Scores)) Mean(SD(Sco)) SD(Mean(Sco)) Max(Mean(Sco)) Min(Mean(Sco))')
print(out_table.numpy())

results_address = logger.get_path() + '/results_by_dist.pickle'
with open(results_address, 'wb') as f:
    pickle.dump(out_table, f)

# Output for plotting: mean loss, max mean loss and min mean loss by distance
with open(logger.get_path()+'/mean_by_dist.pickle', 'wb') as f:
    pickle.dump(mean_by_dist, f)
with open(logger.get_path()+'/max_mean_by_dist.pickle', 'wb') as f:
    pickle.dump(max_mean_by_dist, f)
with open(logger.get_path()+'/min_mean_by_dist.pickle', 'wb') as f:
    pickle.dump(min_mean_by_dist, f)

np.savetxt(logger.get_path() + '/results_by_dist.txt', out_table.numpy(), fmt='%.10f',
           header='NumOfChanges Mean(Scores) Max(Scores) Min(Scores) Mean(Span(Scores)) Mean(SD(Sco)) SD(Mean(Sco)) Max(Mean(Sco)) Min(Mean(Sco))',
           delimiter='   ')

print('-------------------------')
max_mat, max_ind = bfu.get_local_extrema(matrices, mean_per_matrix, get_minima=GET_MINIMA)
print('Number of local ' + ('minima' if GET_MINIMA else 'maxima') +': ' + str(len(max_mat)))
print('Indices: '+ str(max_ind))
print('Mean scores: ' + str(mean_per_matrix[max_ind].numpy()))