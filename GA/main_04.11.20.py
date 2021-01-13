import evaluator as ev
import sys,tools,time
import pickle
from constants import *

logger = tools.Logger()
sys.stdout = logger

# initialize evaluator with given timeseries data
series_address = './data/Voter/timeseries_Voter_grid5x2.pickle'
evaluator = ev.Evaluator(series_address=series_address)

# load ground truth matrix
adj_address = './data/VOTER/edges_Voter_grid5x2.pickle'
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

# uniformly samples a undirected nxn adjacency matrix
def sample_undirected_matrix_uniform(n):
    rand = torch.full((n,n),0.5,device=DEVICE_GEN).bernoulli()
    tools.symmetrize_matrix_(rand)
    return rand

# randomly adds or deletes an edge
def mutate_matrix(matrix):
    x = torch.randint(0,matrix.size()[0])
    y = torch.randint(0,matrix.size()[1])
    matrix[x,y] = 1-matrix[x,y]


# for each entry of matrix, flips the bit with probability p. If sym=True, always returns a symmetric matrix
def mutate_full_matrix(matrix, p, sym=False):
    select_mat = torch.empty((matrix.size()[0], matrix.size()[1]), device = DEVICE_GEN).uniform_() < p
    flipped_mat = 1-matrix
    result = select_mat.to(matrix.dtype)*flipped_mat + (1-select_mat).to(matrix.dtype)*matrix
    if sym:
        tools.symmetrize_matrix_(result)
    return result


# creates a child matrix where each edge is copied from one of the parent matrices with prob 0.5
# mat1, mat2: torch tensors of the same size representing adjacency matrices
def crossover_random(mat1, mat2):
    n = mat1.size()[0]
    m = mat1.size()[1]
    select_mat = torch.empty((n, m), device = DEVICE_GEN).uniform_() > 0.5 # dtype=torch.uint8
    result = select_mat.to(mat1.dtype)*mat1 + (1-select_mat).to(mat1.dtype)*mat2
    return result

# creates two child matrices where one node's connections are taken from the other matrix. -> swap a column & row
def crossover_onenode(mat1, mat2):
    ind = torch.randint(low=0, high=mat1.size()[0], size=(1,), dtype=torch.int32).item()
    res1 = mat1.detach().clone()
    res2 = mat2.detach().clone()
    res1col = res1[ind,:]
    res1row = res1[:,ind]
    res1[ind,:] = res2[ind,:]
    res1[:,ind] = res2[:,ind]
    res2[ind,:] = res1col
    res2[:,ind] = res1row
    return res1, res2

# evaluates the fitness of all individuals in a population
# population: POP_SIZE x n x n tensor of adjacency matrices
# returns: POP_SIZE tensor of non-negative fitness values
def evaluate(population):
    # for debugging: return number of links in the graph
    # return torch.sum(population, dim=(1,2), dtype=torch.float32)
    return evaluator.evaluate_pop(population, use_accu=False)


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

def linear_rank_selection_sorted(sorted_scores):
    ranks = torch.tensor(list(reversed(range(1,POP_SIZE+1))), dtype=torch.float32)
    return ranks / sum(ranks)

#--------------------------------------------------------------------------------------------------------------------
# initialize population
if not CONTINUATION:
    population_list = []
    for i in range(POP_SIZE):
        population_list.append(sample_undirected_matrix_uniform(NUM_NODES))
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

mean_scores = list()
max_scores = list()
tprs = list()
fprs = list()
all_scores = list()
all_populations = list()
start_time = time.process_time()

# evolve for NUM_GEN generations
for i in range(NUM_GEN):
    # evaluate
    scores = evaluate(population)
    all_scores.append(scores)
    all_populations.append(population)

    # collect statistics for round
    argmax_score = torch.argmax(scores)

    (tpr, fpr) = tools.calculate_tpr_fpr(gt_matrix, population[argmax_score])
    mean_score = scores.mean().item()
    max_score = scores[argmax_score].item()
    tprs.append(tpr)
    fprs.append(fpr)
    max_scores.append(max_score)
    mean_scores.append(mean_score)

    # print round summary
    print('Round '+str(i)+' done. -----------------------------------------')
    print('Best score: ' + str(max_score))
    print('Mean score: ' + str(mean_score))
    print('Matrix: \n' + str(population[argmax_score].data.cpu().numpy().astype(int)))
    print('TPR: ' + str(tpr))
    print('FPR: ' + str(fpr))
    print('------------------------------------------------------------')
    logger.flush()

    # select
    # (_, max_ind_nomut) = scores.topk(NUM_NO_MUTATION)  # the best individuals are left untouched
    # (_, max_ind_nocross) = scores.topk(NUM_NO_CROSS)  # the next best ones are only mutated, but not crossed
    (sorted_scores,sorted_indices) = torch.sort(scores, descending=True)
    sorted_pop = population[sorted_indices]
    sorted_probs = linear_rank_selection_sorted(sorted_scores)
    idx = sorted_probs.multinomial(POP_SIZE * 2, replacement=True).view(POP_SIZE, 2)  # sample from selection probabilities

    # new generation
    for i in range(0,POP_SIZE,2):
        # leave the best unchanged
        if i < NUM_NO_MUTATION:
            population[i] = sorted_pop[i]
            population[i+1] = sorted_pop[i+1]
            continue

        if i >= NUM_NO_CROSS:
            # crossover and mutate
            population[i], population[i+1] = crossover_onenode(sorted_pop[idx[i,0]], sorted_pop[idx[i,1]])
            population[i] = mutate_full_matrix(population[i], MUT_PROB, sym=True)
            population[i+1] = mutate_full_matrix(population[i+1], MUT_PROB, sym=True)
        else:
            # just mutate
            population[i] = mutate_full_matrix(sorted_pop[idx[i, 0]], MUT_PROB, sym=True)
            population[i + 1] = mutate_full_matrix(sorted_pop[idx[i + 1, 0]], MUT_PROB, sym=True)


print('Time needed: ' + str(time.process_time() - start_time) + ' sec')
# save population and statistics
path = logger.get_path()
with open(path+'/final_population.pickle', 'wb') as f:
    pickle.dump(population, f)
with open(path+'/all_populations.pickle', 'wb') as f:
    pickle.dump(all_populations, f)
with open(path+'/all_scores.pickle', 'wb') as f:
    pickle.dump(all_scores, f)
with open(path+'/mean_scores.pickle', 'wb') as f:
    pickle.dump(mean_scores, f)
with open(path+'/max_scores.pickle', 'wb') as f:
    pickle.dump(max_scores, f)
with open(path+'/tprs.pickle', 'wb') as f:
    pickle.dump(tprs, f)
with open(path+'/fprs.pickle', 'wb') as f:
    pickle.dump(fprs, f)