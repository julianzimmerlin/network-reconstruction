import evaluator as ev, tools
import pickle, sys
from constants import *

logger = tools.Logger('brute_force_analysis_logs')
sys.stdout = logger

# load ground truth mat
adj_address = './data/Voter/edges_bull.pickle'
with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')

print(gt_matrix)

# calculates number of flipped bits between mat1 and mat2
def calc_num_flips(mat1, mat2):
    num = 0
    for i in range(5):
        for j in range(i,5):
            if not torch.eq(mat1[i,j], mat2[i,j]):
                num += 1
    return num

# returns a list which contains in position i a list of all indices of matrices that have hamming distance i to gt_matrix
def get_indices_by_dist(matrices, selfloops=False):
    inds_by_dist = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()] if not selfloops else \
        [list(), list(), list(), list(), list(), list()]
    for i in range(len(matrices)):
        flips = calc_num_flips(gt_matrix, matrices[i])
        inds_by_dist[flips].append(i)
    return inds_by_dist

def get_local_maxima(matrices, mean_scores):
    max_scores, max_indices = list(), list()
    print('evaluating local maxima')
    for i in range(len(matrices)):
        is_max = True
        for j in range(len(matrices)):
            if (calc_num_flips(matrices[i], matrices[j]) == 1):
                if (mean_scores[j] >= mean_scores[i]):
                    is_max = False
                    break
        if is_max:
            max_scores.append(matrices[i])
            max_indices.append(i)
    return max_scores, max_indices


# ----------------------------------------------

source = './brute_force_logs/voter_bull_selfloops'
matrices_address = source + '/matrices.pickle'
with open(matrices_address, 'rb') as f:
    matrices = pickle.load(f)
scores_address = source + '/scores.pickle'
with open(scores_address, 'rb') as f:
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
(best_matrices_scores, best_indices) = mean_per_matrix.topk(20)
num_changes_for_topk = torch.zeros(len(best_indices))
for i in range(len(best_indices)):
    num_changes_for_topk[i] = calc_num_flips(gt_matrix, matrices[best_indices[i]])

# compare by hamming distance to gt_matrix
inds_by_dist = get_indices_by_dist(matrices, selfloops=True)
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
    inds = inds_by_dist[i]
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

np.savetxt(logger.get_path() + '/results_by_dist.txt', out_table.numpy(), fmt='%.10f',
           header='NumOfChanges Mean(Scores) Max(Scores) Min(Scores) Mean(Span(Scores)) Mean(SD(Sco)) SD(Mean(Sco)) Max(Mean(Sco)) Min(Mean(Sco))',
           delimiter='   ')

print('-------------------------')
max_mat, max_ind = get_local_maxima(matrices, mean_per_matrix)
print('Number of local maxima: ' + str(len(max_mat)))
print('Indices: '+ str(max_ind))
print('Mean scores: ' + str(mean_per_matrix[max_ind].numpy()))