import torch
import hashlib

# calculates fpr and tpr of a matrix compared to the ground truth adjacency matrix
def calculate_tpr_fpr(ground_truth, mat, skip_diag=True):
    matrix = mat.cpu()
    tp = 0
    p = 0
    n = 0
    fp = 0
    for i in range(ground_truth.size()[0]):
        for j in range(ground_truth.size()[1]):
            if skip_diag and i == j:
                continue
            if(ground_truth[i,j] == 0):
                n += 1
                if(matrix[i,j] == 1):
                    fp += 1
            else:
                p += 1
                if(matrix[i,j] == 1):
                    tp += 1
    return (tp / p, fp / n)

# calculates accuracy of a matrix compared to the ground truth adjacency matrix
def calculate_accuracy(ground_truth, mat, skip_diag=True):
    num_nodes = ground_truth.size()[0]
    mask = (1-torch.eye(num_nodes, num_nodes)).to(torch.uint8) \
        if skip_diag else torch.ones(num_nodes, num_nodes)
    err_ml_nodiag = torch.sum(torch.abs(mat.masked_select(mask) - ground_truth.masked_select(mask)))
    accnet_nodiag = 1 - err_ml_nodiag.item() / ground_truth.masked_select(mask).numel()
    return accnet_nodiag

def hash_tensor(matrix):
    string = ''
    triu_indices = torch.triu_indices(row=matrix.size()[0], col=matrix.size()[0], offset=1)
    for entry in matrix[triu_indices[0], triu_indices[1]]:
        string += str(entry.to(torch.int).item())

    m = hashlib.shake_128()
    m.update(bytes(string, 'utf-8'))
    return m.hexdigest(8)

# converts a matrix id (integer between 0 and 2^(num_nodes*(num_nodes-1) // 2) to the corresponding matrix of size num_nodes
# def id2matrix(id, num_nodes):
#     num_entries = num_nodes*(num_nodes-1) // 2
#     bin_id = list(('{0:0' + str(num_entries) + 'b}').format(id)) # convert to binary
#     bin_id_tensor = torch.tensor([int(num) for num in bin_id], dtype=torch.float32)
#     mat = torch.zeros(num_nodes, num_nodes)
#     triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1)
#     mat[triu_indices[0], triu_indices[1]] = bin_id_tensor
#     symmetrize_matrix_(mat, take_mean=False)
#     return mat
# 
# def matrix2id(matrix):
#     triu_indices = torch.triu_indices(row=matrix.size()[0], col=matrix.size()[0], offset=1)
#     triu_vec = matrix[triu_indices[0], triu_indices[1]]
#     bits = matrix.size()[0] * (matrix.size()[0]-1) // 2
#     mask = 2 ** torch.arange(bits-1, -1, -1).to(triu_vec.device, triu_vec.dtype)
#     return torch.sum(mask * triu_vec, -1).to(torch.long).item()

# makes a matrix symmetric by copying all values from the upper half to the lower half or taking the mean values
def symmetrize_matrix_(mat, take_mean=False):
    n = mat.size()[0]
    for i in range(n):
        for j in range(i+1,n):
            if take_mean:
                val = (mat[j,i]+mat[i,j]) / 2
                mat[j,i] = mat[i,j] = val
            else:
                mat[j,i] = mat[i,j]

# uniformly samples a undirected nxn adjacency matrix
def sample_undirected_matrix_uniform(n):
    rand = torch.full((n,n),0.5).bernoulli()
    for i in range(n):
        rand[i,i] = 0
    symmetrize_matrix_(rand)
    rand.requires_grad_(True)
    return rand

# this function indicates whether an optimization algorithm has converged
# by looking at the differences in losses at consecutive steps
def has_converged(lo_list):
    if len(lo_list) < 10:
        return False
    losses = torch.stack(lo_list)
    diffs = abs(losses[1:]-losses[:-1])
    if (diffs[-9:] < 4e-5).all():
        return True
    else:
        return False

def get_all_neighbors(matrix):
    neighbors = list()
    for i in range(matrix.size()[0]):
       for j in range(i+1, matrix.size()[1]):
           neighbor = matrix.detach().clone()
           neighbor[i,j] = 1 - neighbor[i,j]
           neighbor[j,i] = 1 - neighbor[j,i]
           neighbor.requires_grad_(True)
           neighbors.append(neighbor)
    return neighbors


def calc_mutation_order_gradient(matrix):
    symmetrize_matrix_(matrix.grad, take_mean=True)
    minus_grad = -1 * matrix.grad
    flipped_matrix = 1 - matrix.detach()
    gradient_partially_flipped = flipped_matrix * minus_grad + matrix.detach() * matrix.grad
    tril_selection_mat = torch.ones_like(gradient_partially_flipped).tril(diagonal=0) == 1
    gradient_partially_flipped[tril_selection_mat] = -torch.ones(matrix.size()[0] * (matrix.size()[0] + 1) // 2)
    top_entries, top_indices = gradient_partially_flipped.view(-1).topk(matrix.size()[0] * (matrix.size()[0]-1) // 2)
    top_indices_rows = top_indices // matrix.size()[1]
    top_indices_cols = top_indices % matrix.size()[0]
    top_ind_stacked = torch.stack((top_indices_rows, top_indices_cols), dim=-1)
    return top_ind_stacked

def calc_mutation_order_evalepoch(matrix, dyn_learner, evaluator):
    indices = list()
    losses = list()
    for i in range(matrix.size()[0]):
        for j in range(matrix.size()[1]):
            if j <= i:
                continue
            neighbor = matrix.detach().clone()
            neighbor[i,j] = 1 - neighbor[i,j]
            neighbor[j,i] = 1 - neighbor[j,i]
            _, loss = evaluator.evaluate_individual_no_training(neighbor, dyn_learner, False)
            indices.append(torch.tensor([i,j]))
            losses.append(torch.tensor(loss))
    indices_tensor = torch.stack(indices)
    losses_tensor = torch.stack(losses)
    best_losses, best_indices = losses_tensor.sort()
    indices_tensor = indices_tensor[best_indices]
    return indices_tensor