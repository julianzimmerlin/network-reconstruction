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
def sample_undirected_matrix_uniform(n, requires_grad=True):
    rand = torch.full((n,n),0.5).bernoulli()
    for i in range(n):
        rand[i,i] = 0
    symmetrize_matrix_(rand)
    rand.requires_grad_(requires_grad)
    return rand

# this function indicates whether an optimization algorithm has converged
# by looking at the differences in losses at consecutive steps
# threshold: 4e-5 works for sis
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
        for j in range(i+1, matrix.size()[1]):
            neighbor = matrix.detach().clone()
            neighbor[i,j] = 1 - neighbor[i,j]
            neighbor[j,i] = 1 - neighbor[j,i]
            loss = evaluator.evaluate_individual_no_training(neighbor, dyn_learner)
            indices.append(torch.tensor([i,j]))
            losses.append(loss.mean())
    indices_tensor = torch.stack(indices)
    losses_tensor = torch.stack(losses)
    best_losses, best_indices = losses_tensor.sort()
    indices_tensor = indices_tensor[best_indices]
    return indices_tensor

def calc_edge_mutation_probs_gradient(matrix):
    symmetrize_matrix_(matrix.grad, take_mean=True)
    minus_grad = -1 * matrix.grad
    flipped_matrix = 1 - matrix.detach()
    gradient_partially_flipped = flipped_matrix * minus_grad + matrix.detach() * matrix.grad
    mask = torch.eye(gradient_partially_flipped.size()[0], gradient_partially_flipped.size()[1]).byte()
    gradient_partially_flipped.masked_fill_(mask, 0)
    scaled = gradient_partially_flipped / gradient_partially_flipped.max()
    scaled[scaled<0] = 0
    return scaled

def calc_edge_mutation_probs_evalepoch(matrix, dyn_learner, evaluator):
    matrix_loss = evaluator.evaluate_individual_no_training(matrix, dyn_learner)
    lossdifferences = torch.zeros_like(matrix)
    for i in range(matrix.size()[0]):
        for j in range(i+1, matrix.size()[1]):
            neighbor = matrix.detach().clone()
            neighbor[i, j] = 1 - neighbor[i, j]
            neighbor[j, i] = 1 - neighbor[j, i]
            neighbor_loss = evaluator.evaluate_individual_no_training(neighbor, dyn_learner)
            lossdifferences[i,j] = matrix_loss-neighbor_loss
    scaled = lossdifferences / lossdifferences.max()
    scaled[scaled < 0] = 0
    return scaled

def calc_edge_mutation_probs_evalepoch_nodewise(matrix, dyn_learner, evaluator):
    matrix_loss = evaluator.evaluate_individual_no_training(matrix, dyn_learner)
    lossdifferences = torch.zeros_like(matrix)
    for i in range(matrix.size()[0]):
        for j in range(i+1, matrix.size()[1]):
            neighbor = matrix.detach().clone()
            neighbor[i, j] = 1 - neighbor[i, j]
            neighbor[j, i] = 1 - neighbor[j, i]
            neighbor_loss = evaluator.evaluate_individual_no_training(neighbor, dyn_learner)
            lossdifferences[i,j] = matrix_loss-neighbor_loss
    symmetrize_matrix_(lossdifferences, take_mean=False)
    probs = torch.zeros_like(matrix)
    for i in range(matrix.size()[0]):
        probs[i] = lossdifferences[i] / lossdifferences[i].max()
    symmetrize_matrix_(probs, take_mean=True)
    probs[probs<0] = 0
    return probs

# -------------------------- now the function as presented in the thesis

def calc_S_grad(matrix):
    raw_gradient = matrix.grad.detach().clone()
    minus_grad = -1 * raw_gradient
    flipped_matrix = 1 - matrix.detach()
    S_asym = flipped_matrix * minus_grad + matrix.detach() * raw_gradient
    S_grad = S_asym + S_asym.transpose(0,1)
    S_grad[torch.eye(S_grad.size()[0]).byte()] = 0
    if torch.all(S_grad<=0):
        print('S_GRAD IS ALL NEGATIVE!')
    return S_grad

def calc_S_eval(matrix, dyn_learner, evaluator,  matrix_loss=None):
    if matrix_loss is None:
        matrix_loss = evaluator.evaluate_individual_no_training(matrix, dyn_learner)
    indices = list()
    losses = list()
    S = torch.zeros_like(matrix)
    for i in range(matrix.size()[0]):
        for j in range(i + 1, matrix.size()[1]):
            neighbor = matrix.detach().clone()
            neighbor[i, j] = 1 - neighbor[i, j]
            neighbor[j, i] = 1 - neighbor[j, i]
            loss = evaluator.evaluate_individual_no_training(neighbor, dyn_learner)
            S[i,j] = matrix_loss.mean() - loss.mean()
            S[j,i] = matrix_loss.mean() - loss.mean()
    if torch.all(S<=0):
        print('S IS ALL NEGATIVE!')
    return S
def dynamic_step_probabilities_eval(matrix, dyn_learner, evaluator, matrix_loss=None):
    S_eval = calc_S_eval(matrix, dyn_learner, evaluator,  matrix_loss=None)
    return dynamic_step_probabilities(S_eval)

def dynamic_step_probabilities_grad(matrix):
    S_grad = calc_S_grad(matrix)
    return dynamic_step_probabilities(S_grad)

def dynamic_step_probabilities_random(matrix):
    N = matrix.size()[0]
    S_uni = 1 - torch.eye(matrix.size()[0])
    return 1/(N-1) * S_uni

def dynamic_step_probabilities(S):
    if S.max()<=0:
        return torch.zeros_like(S)
    probs = S / S.max()
    probs[probs < 0] = 0
    return probs

def single_step_probabilities_random(matrix):
    S_uni = 1 - torch.eye(matrix.size()[0])
    sum = torch.sum(S_uni) / 2
    return S_uni / sum

def single_step_probabilities_eval(matrix, dyn_learner, evaluator, matrix_loss=None):
    S_eval = calc_S_eval(matrix, dyn_learner, evaluator, matrix_loss=None)
    return single_step_probabilities(S_eval)

def single_step_probabilities_grad(matrix):
    S_grad = calc_S_grad(matrix)
    return single_step_probabilities(S_grad)

def single_step_probabilities(S):
    S[S<0] = 0
    sum = torch.sum(S) / 2
    if sum <=0:
        # return single_step_probabilities_random(S)
        return None
    probs = S / sum
    return probs

def exec_dynamic_step_grad(matrix):
    probs = dynamic_step_probabilities_grad(matrix)
    return exec_dynamic_step(matrix, probs)

def exec_dynamic_step_eval(matrix, dyn_learner, evaluator, matrix_loss=None):
    probs = dynamic_step_probabilities_eval(matrix, dyn_learner, evaluator,  matrix_loss=None)
    return exec_dynamic_step(matrix, probs)

def exec_dynamic_step_random(matrix):
    probs = dynamic_step_probabilities_random(matrix)
    return exec_dynamic_step(matrix, probs)

def exec_dynamic_step(matrix, probs):
    new_cand = matrix.detach().clone()
    # flip each edge with the the respective probability in probs
    indices = list()
    for i in range(probs.size()[0]):
        for j in range(i + 1, probs.size()[1]):
            if torch.rand(1, 1).item() < probs[i, j]:
                new_cand[i, j] = 1 - new_cand[i, j]
                new_cand[j, i] = 1 - new_cand[j, i]
                indices.append(torch.tensor([i,j]))

    #print('Candidate: ' + ut.hash_tensor(new_cand) + '. Changes:' + str(num_changes))
    return new_cand, indices

def exec_single_step_grad(matrix):
    probs = single_step_probabilities_grad(matrix)
    return exec_single_step(matrix, probs)

def exec_single_step_eval(matrix, dyn_learner, evaluator, matrix_loss=None):
    probs = single_step_probabilities_eval(matrix, dyn_learner, evaluator,  matrix_loss=None)
    return exec_single_step(matrix, probs)

def exec_single_step_random(matrix):
    probs = single_step_probabilities_random(matrix)
    return exec_single_step(matrix, probs)

def exec_single_step(matrix, probs):
    if probs is None: # Fallback behaviour: return same matrix again
        return matrix.detach().clone(), []
    probs_triu = torch.triu(probs, diagonal=1)
    probs_vec = probs_triu.view(-1)  # vectorize
    sample = probs_vec.multinomial(num_samples=1)
    index = torch.cat([sample // matrix.size()[0], sample % matrix.size()[0]])

    result = matrix.detach().clone()
    result[list(index)] = 1 - result[list(index)]
    result[list(index.flip(dims=(0,)))] = 1 - result[list(index.flip(dims=(0,)))]
    return result, [index]
