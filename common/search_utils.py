import torch
import utils as ut

# use guided mutation. changes exactly one entry of the matrix
def mutate_matrix_guided(matrix, dyn_learner, evaluator):
    #probs = calc_guided_mutation_probs(matrix, sym=True, softmax_factor= 1 / TEMPERATURE)
    #print(probs.cpu().numpy())
    #probs_vec = probs.view(-1)  # vectorize
    #sample = probs_vec.multinomial(num_samples=1)
    #index = torch.stack([sample // matrix.size()[0], sample % matrix.size()[0]])

    indices_order = ut.calc_mutation_order_evalepoch(matrix, dyn_learner, evaluator)
    n = matrix.size()[0]
    probs = torch.tensor([1 / (2**i) for i in range(n*(n-1)//2)])
    sample = probs.multinomial(num_samples=1)
    index = indices_order[sample,:].squeeze()

    result = matrix.detach().clone()
    result[list(index)] = 1 - result[list(index)]
    result[list(index.flip(dims=(0,)))] = 1 - result[list(index.flip(dims=(0,)))]
    result.requires_grad_(True)
    return result, index

def double_mutation(matrix, dyn_learner, evaluator):
    intermediate, inter_index = mutate_matrix_guided(matrix, dyn_learner, evaluator)
    indices_order = ut.calc_mutation_order_evalepoch(intermediate, dyn_learner, evaluator)
    index = indices_order[0,:].squeeze()

    result = intermediate.detach().clone()
    result[list(index)] = 1 - result[list(index)]
    result[list(index.flip(dims=(0,)))] = 1 - result[list(index.flip(dims=(0,)))]
    result.requires_grad_(True)

    print('first: ' + str(inter_index.tolist()) + ', second: ' + str(index.tolist()), end='. ')
    if (inter_index == index).all().item():
        print('Single mut.')
    else:
        print('1:add' if matrix[list(inter_index)] == 0 else '1:del', end=', ')
        print('2:add' if intermediate[list(index)] == 0 else '2:del', end='. ')
        if any([a == b for a in index.flatten() for b in inter_index.flatten()]):
            print('Same node.')
        else:
            print(' ')

    if (inter_index == index).all().item(): # if double mutation would just revert the mutation, return intermediate
        return intermediate
    return result


def calc_guided_mutation_probs(matrix, softmax_factor=1.):
    ut.symmetrize_matrix_(matrix.grad, take_mean=True)
    minus_grad = -1 * matrix.grad
    flipped_matrix = 1 - matrix.detach()
    gradient_partially_flipped = flipped_matrix * minus_grad + matrix.detach() * matrix.grad
    #print(gradient_partially_flipped)
    softmax = torch.nn.Softmax(dim=0)
    triu_selection_mat = torch.ones_like(gradient_partially_flipped).triu(diagonal=1) == 1
    triu_vec = gradient_partially_flipped[triu_selection_mat]  # get upper triangular matrix as vector
    probs_vec = softmax(triu_vec * softmax_factor)
    probs = torch.zeros_like(gradient_partially_flipped)
    probs[triu_selection_mat] = probs_vec
    return probs