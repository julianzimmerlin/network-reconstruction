import torch

# calculates number of flipped bits between mat1 and mat2
def calc_num_flips(mat1, mat2):
    num = 0
    for i in range(5):
        for j in range(i,5):
            if not torch.eq(mat1[i,j], mat2[i,j]):
                num += 1
    return num

# returns a list which contains in position i a list of all indices of matrices that have hamming distance i to ref_matrix
def get_indices_by_dist(ref_matrix, matrices):
    inds_by_dist = list()
    num_nodes = ref_matrix.size()[0]
    for _ in range(num_nodes * (num_nodes-1) // 2 + 1):
        inds_by_dist.append(list())
    for i in range(len(matrices)):
        flips = calc_num_flips(ref_matrix, matrices[i])
        inds_by_dist[flips].append(i)
    return inds_by_dist

def get_local_extrema(matrices, mean_scores, get_minima):
    max_scores, max_indices = list(), list()
    print('evaluating local  ' + ('minima' if get_minima else 'maxima'))
    for i in range(len(matrices)):
        is_max = True
        for j in range(len(matrices)):
            if (calc_num_flips(matrices[i], matrices[j]) == 1):
                if get_minima:
                    if (mean_scores[j] < mean_scores[i]):
                        is_max = False
                        break
                else:
                    if (mean_scores[j] > mean_scores[i]):
                        is_max = False
                        break
        if is_max:
            max_scores.append(matrices[i])
            max_indices.append(i)
    return max_scores, max_indices
