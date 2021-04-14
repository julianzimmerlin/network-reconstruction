import torch
import evaluator as ev
import sys,tools
import pickle
from constants import *

logger = tools.Logger()
sys.stdout = logger

# initialize evaluator with given timeseries data
series_address = './data/Voter/timeseries_BA_15_2.pickle'
evaluator = ev.Evaluator(series_address=series_address)

# load ground truth matrix
adj_address = './data/VOTER/edges_BA_15_2.pickle'
with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')


# samples a directed nxn adjacency matrix with k edges (actually it can also have less links by chance)
def sample_directed_matrix_k_edges(n, k):
    mat = torch.zeros((n,n), device=DEVICE_GEN)
    positions = torch.randint(low=0,high=n*n, size=(k,),dtype=torch.uint8)
    for i in range(k):
        x = positions[i] // n
        y = positions[i] % n
        mat[x.item(),y.item()] = 1
    return mat


# randomly adds or deletes an edge
def mutate_matrix(matrix):
    x = torch.randint(0,matrix.size()[0])
    y = torch.randint(0,matrix.size()[1])
    matrix[x,y] = 1-matrix[x,y]


# for each entry of matrix, flips the bit with probability p.
def mutate_full_matrix(matrix, p):
    select_mat = torch.empty((matrix.size()[0], matrix.size()[1]), device = DEVICE_GEN).uniform_() < p
    flipped_mat = 1-matrix
    result = select_mat.to(matrix.dtype)*flipped_mat + (1-select_mat).to(matrix.dtype)*matrix
    return result


# creates a child matrix where each edge is copied from one of the parent matrices with prob 0.5
# mat1, mat2: torch tensors of the same size representing adjacency matrices
def crossover_random(mat1, mat2):
    n = mat1.size()[0]
    m = mat1.size()[1]
    select_mat = torch.empty((n, m), device = DEVICE_GEN).uniform_() > 0.5 # dtype=torch.uint8
    result = select_mat.to(mat1.dtype)*mat1 + (1-select_mat).to(mat1.dtype)*mat2
    return result


# evaluates the fitness of all individuals in a population
# population: POP_SIZE x n x n tensor of adjacency matrices
# returns: POP_SIZE tensor of non-negative fitness values
def evaluate(population):
    # for debugging: return number of links in the graph
    # return torch.sum(population, dim=(1,2), dtype=torch.float32)
    return evaluator.evaluate_pop(population)


# uses a softmax of the scores to determine selection probability
# scores: a (POP_SIZE,) tensor with fitness scores for each individual
# returns: a (POP_SIZE,) tensor with selection probabilities for each individual
def softmax_selection(scores):
    return torch.exp(scores) / sum(torch.exp(scores))


# simple normalization of non-negative scores
def roulette_wheel_selection(scores):
    return scores / sum(scores)


# The best individual gets p = POP_SIZE / (n(n-1)), next one p = (POP_SIZE-1) / (n(n-1)), ... , last one p = 1 / n(n-1)
def linear_rank_selection(scores):
    (_,indices) = torch.sort(scores, descending=True)
    ranks = torch.zeros_like(scores)
    cur_rank = POP_SIZE
    for i in indices:
        ranks[i] = cur_rank
        cur_rank -= 1
    return ranks / sum(ranks)

#--------------------------------------------------------------------------------------------------------------------
# initialize population
if not CONTINUATION:
    population_list = []
    for i in range(POP_SIZE):
        population_list.append(sample_directed_matrix_k_edges(NUM_NODES, 5))
    population = torch.stack(population_list)
else:
    with open(CONT_FILE, 'rb') as f:
        population = pickle.load(f)

print('NUM_GEN: ' + str(NUM_GEN))
print('POP_SIZE: ' + str(POP_SIZE))
print('MUT_PROB: ' + str(MUT_PROB))
print('NUM_NO_MUTATION: ' + str(NUM_NO_MUTATION))
print('NUM_NO_CROSS: ' + str(NUM_NO_CROSS))
print('NUM_NODES: ' + str(NUM_NODES))
print('CONTINUATION: ' + str(CONTINUATION))
print('--------------------- Start training ------------------------')

# evolve for NUM_GEN generations
for i in range(NUM_GEN):
    # evaluate
    scores = evaluate(population)

    # print output
    argmax_score = torch.argmax(scores)
    (tpr, fpr) = tools.calculate_tpr_fpr(gt_matrix, population[argmax_score])

    print('Round '+str(i)+' done. -----------------------------------------')
    print('Best score: ' + str(scores[argmax_score].item()))
    print('Mean score: ' + str(scores.mean().item()))
    print('Matrix: \n' + str(population[argmax_score].data.cpu().numpy().astype(int)))
    print('TPR: ' + str(tpr))
    print('FPR: ' + str(fpr))
    print('------------------------------------------------------------')
    logger.flush()

    # select
    (_, max_ind_nomut) = scores.topk(NUM_NO_MUTATION)  # the best individuals are left untouched
    (_, max_ind_nocross) = scores.topk(NUM_NO_CROSS)  # the next best ones are only mutated, but not crossed
    probs = linear_rank_selection(scores)
    # probs = softmax_selection(scores)
    idx = probs.multinomial(POP_SIZE * 2, replacement=True).view(POP_SIZE, 2)  # sample from selection probabilities


    # new generation
    for i in range(POP_SIZE):
        if i in max_ind_nomut: continue
        # crossover
        if i not in max_ind_nocross:
            population[i] = crossover_random(population[idx[i,0]], population[idx[i,1]])
        # mutate
        # if rng.random()<0.5:
        #    mutate_matrix(population[i])
        population[i] = mutate_full_matrix(population[i], MUT_PROB)


# save population
path = logger.get_path()
with open(path+'/population.pickle', 'wb') as f:
    pickle.dump(population, f)


