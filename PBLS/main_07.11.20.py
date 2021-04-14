import evaluator as ev
import sys,tools,time
import pickle
from constants import *

logger = tools.Logger()
sys.stdout = logger
path = logger.get_path()

# initialize evaluator with given timeseries data
series_address = './data/Voter/timeseries_bull_3000.pickle'
evaluator = ev.Evaluator(series_address=series_address)

# load ground truth matrix
adj_address = './data/VOTER/edges_bull.pickle'
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


def calc_print_save_statistics(population, scores, all_populations, all_scores, mean_scores, max_scores, tprs, fprs):
    # collect statistics for round
    argmax_score = torch.argmax(scores)

    (tpr, fpr) = tools.calculate_tpr_fpr(gt_matrix, population[argmax_score])
    mean_score = scores.mean().item()
    max_score = scores[argmax_score].item()
    tprs.append(tpr)
    fprs.append(fpr)
    max_scores.append(max_score)
    mean_scores.append(mean_score)
    all_scores.append(scores)
    all_populations.append(population)

    # save statistics
    with open(path + '/final_population.pickle', 'wb') as f:
        pickle.dump(population, f)
    with open(path + '/all_populations.pickle', 'wb') as f:
        pickle.dump(all_populations, f)
    with open(path + '/all_scores.pickle', 'wb') as f:
        pickle.dump(all_scores, f)
    with open(path + '/mean_scores.pickle', 'wb') as f:
        pickle.dump(mean_scores, f)
    with open(path + '/max_scores.pickle', 'wb') as f:
        pickle.dump(max_scores, f)
    with open(path + '/tprs.pickle', 'wb') as f:
        pickle.dump(tprs, f)
    with open(path + '/fprs.pickle', 'wb') as f:
        pickle.dump(fprs, f)

    # print round summary
    print('Round ' + str(i) + ' done. -----------------------------------------')
    print('Best score: ' + str(max_score))
    print('Mean score: ' + str(mean_score))
    print('Matrix: \n' + str(population[argmax_score].data.cpu().numpy().astype(int)))
    print('TPR: ' + str(tpr))
    print('FPR: ' + str(fpr))
    print('------------------------------------------------------------')
    logger.flush()

#---------------------------------------------------------------------NOW WITH SELECTION PRESSURE
# changes since 500gen experiment on ba20: truncation select instead of linear rank, don't optimize in evaluation epoch anymore
print('NUM_GEN: ' + str(NUM_GEN))
print('POP_SIZE: ' + str(POP_SIZE))
print('MUT_PROB: ' + str(MUT_PROB))
print('CROSS_PROB: ' + str(CROSS_PROB))
print('NUM_NODES: ' + str(NUM_NODES))
print('CONTINUATION: ' + str(CONTINUATION))
print('--------------------- Start training ------------------------')

# initialization
if not CONTINUATION:
    mean_scores = list()
    max_scores = list()
    tprs = list()
    fprs = list()
    all_scores = list()
    all_populations = list()
    population_list = []
    for i in range(POP_SIZE):
        population_list.append(sample_undirected_matrix_uniform(NUM_NODES))
    population = torch.stack(population_list)#
    scores = evaluate(population)
else:  # load statistics from previous run
    with open(CONT_PATH + '/final_population.pickle', 'rb') as f:
        population = pickle.load(f)
    with open(CONT_PATH + '/all_populations.pickle', 'rb') as f:
        all_populations = pickle.load(f)
    with open(CONT_PATH + '/all_scores.pickle', 'rb') as f:
        all_scores = pickle.load(f)
    with open(CONT_PATH + '/mean_scores.pickle', 'rb') as f:
        mean_scores = pickle.load(f)
    with open(CONT_PATH + '/max_scores.pickle', 'rb') as f:
        max_scores = pickle.load(f)
    with open(CONT_PATH + '/tprs.pickle', 'rb') as f:
        tprs = pickle.load(f)
    with open(CONT_PATH + '/fprs.pickle', 'rb') as f:
        fprs = pickle.load(f)
    scores = all_scores[-1]
start_time = time.process_time()

# evolve for NUM_GEN generations
for i in range(NUM_GEN):
    if not (i==0 and CONTINUATION):
        calc_print_save_statistics(population, scores, all_populations, all_scores, mean_scores, max_scores, tprs, fprs)

    # select
    # probs = linear_rank_selection(scores)
    # idx = probs.multinomial(NEWPOP_SIZE, replacement=True)  # sample from selection probabilities
    idx = torch.randint(high=POP_SIZE, size=(NEWPOP_SIZE,), dtype=torch.long)

    # new generation
    newpop = torch.zeros(NEWPOP_SIZE, NUM_NODES, NUM_NODES)
    for i in range(0, NEWPOP_SIZE, 2):
        # crossover
        if torch.randn(1).item() < CROSS_PROB:  # THIS IS A BUG, SHOULD BE rand, not randn
            newpop[i], newpop[i+1] = crossover_onenode(population[idx[i]], population[idx[i+1]])
        else:
            newpop[i] = population[idx[i]]
            newpop[i+1] = population[idx[i+1]]
        # mutate
        newpop[i] = mutate_full_matrix(newpop[i], MUT_PROB, sym=True)
        newpop[i+1] = mutate_full_matrix(newpop[i+1], MUT_PROB, sym=True)

    # evaluate old and new population and merge
    scores = evaluate(population)
    newscores = evaluate(newpop)
    # update population
    fullpop = torch.cat((population, newpop))
    fullscores = torch.cat((scores, newscores))
    (scores, best_individual_indices) = fullscores.topk(POP_SIZE)
    population = fullpop[best_individual_indices]


calc_print_save_statistics(population, scores, all_populations, all_scores, mean_scores, max_scores, tprs, fprs)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')
