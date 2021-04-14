import evaluator as ev, tools
import pickle, sys
from constants import *

logger = tools.Logger('objective_test_logs')
sys.stdout = logger

# initialize evaluator with given timeseries
series_address = './data/SIS/timeseries_bull_5k_0.5_0.5.pickle'
evaluator = ev.Evaluator(series_address=series_address)

# load ground truth mat
adj_address = './data/VOTER/edges_bull.pickle'
with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')

print(gt_matrix)

# return a list with length num_matrices of matrices that have hamming distance num_flips to gt_mat
def get_matrices(gt_mat, num_flips, num_matrices):
    matrices = []
    while len(matrices) < num_matrices:
        indices = []
        mat = gt_mat.detach().clone()
        while len(indices) < num_flips:
            (i, j) = get_random_index()
            if (i, j) not in indices:
                mat[i, j] = 1 - mat[i, j]
                mat[j, i] = 1 - mat[j, i]
                indices.append((i, j))
        if not any([torch.eq(mat, m).all() for m in matrices]):
            matrices.append(mat)
    return matrices


# returns a random index in the upper half of a 5x5 matrix
def get_random_index():
    i = 0
    j = 0
    while i == j:
        num = int(torch.randint(25, (1,)).item())
        i = num // 5
        j = num % 5
    return (i, j) if i < j else (j, i)


# ----------------------------------------------------------------
num_matrices = 10
num_runs = 50
max_flips = 9
print('SEED: {}'.format(SEED))
print('max_flips: {}'.format(max_flips))
print('num_matrices: {}'.format(num_matrices))
print('num_runs: {}'.format(num_runs))

scores = torch.zeros(max_flips, num_matrices, num_runs)

# evaluate ground truth
gt_scores = torch.zeros(num_runs)
for i in range(num_runs):
    sc = evaluator.evaluate_individual(gt_matrix)
    gt_scores[i] = sc
gt_scores = gt_scores.repeat(1,num_matrices,1)

scores = torch.cat((gt_scores, scores))
# evaluate neighbors
for fl in range(1, max_flips+1):
    matrices = get_matrices(gt_matrix, fl, num_matrices)
    for mat_ind in range(len(matrices)):
        for run in range(num_runs):
            print('evaluating: flip ' + str(fl) + ', matrix ' + str(mat_ind) + ', run ' + str(run) + '. ', end='')
            scores[fl][mat_ind][run] = evaluator.evaluate_individual(matrices[mat_ind])

scores_address = logger.get_path() + '/scores.pickle'
with open(scores_address, 'wb') as f:
    pickle.dump(scores, f)

print('-------------- RESULTS -----------------')
means_per_matrix = scores.mean(dim=2)
stdev_per_matrix = scores.std(dim=2)
max_per_matrix,_ = scores.max(dim=2)
min_per_matrix,_ = scores.min(dim=2)

averages_per_flip = means_per_matrix.mean(dim=1)
max_per_flip,_ = max_per_matrix.max(dim=1)
min_per_flip,_ = min_per_matrix.min(dim=1)
mean_stdev_per_flip = stdev_per_matrix.mean(dim=1)
stdev_mean_per_flip = means_per_matrix.std(dim=1)
max_mean_per_flip,_ = means_per_matrix.max(dim=1)
min_mean_per_flip,_ = means_per_matrix.min(dim=1)

Num_changes = torch.tensor(list(range(0,max_flips+1)), dtype=torch.float32)
out_table = torch.stack([Num_changes, averages_per_flip, max_per_flip, min_per_flip, mean_stdev_per_flip, stdev_mean_per_flip, max_mean_per_flip, min_mean_per_flip], dim=1)
print('NumOfChanges Mean(Scores) Max(Scores) Min(Scores) Mean(SD(Scores)) SD(Mean(Sco)) Max(Mean(Sco)) Min(Mean(Sco))')
np.set_printoptions(precision=10, suppress=True, linewidth=200)
print(out_table.numpy())

results_address = logger.get_path() + '/results.pickle'
with open(results_address, 'wb') as f:
    pickle.dump(out_table, f)

np.savetxt(logger.get_path() + '/results.txt', out_table.numpy(), fmt='%.10f',
           header='Hamming Dist. Mean(Scores) Max(Scores) Min(Scores) Mean(SD(Scores)) Max(Mean(Scores)) Min(Mean(Scores))',
           delimiter='   ')



