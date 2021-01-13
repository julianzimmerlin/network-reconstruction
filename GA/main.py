import sys, pickle
import torch
import evaluator as ev
import logger as lo
import numpy as np
import tracker as tr
import utils as ut

SEED = 123421
SERIES_ADDRESS = '../data/SIS_FIXED/timeseries_bull_1k_0.2_0.2.pickle'
ADJ_ADDRESS = '../data/SIS_FIXED/edges_bull.pickle'
BATCH_SIZE = 5000
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS = 160
DETECT_EARLY_CONVERGENCE = True
USE_GUIDED_MUTATIONS = True
POP_SIZE = 2
NEWPOP_SIZE = 2
NUM_GEN = 500
CROSS_PROB = 0.
TEMPERATURE = 1 / 2500
USE_OLD_DISCRETE_FORMAT = True
CONTINUATION = False
CONT_ADDRESS = r'D:\Uni\BA\ColabOutputs\ba20\2020-12-10T22_50_22.113418'

logger = lo.Logger('GA_logs')
sys.stdout = logger
torch.manual_seed(SEED)
np.random.seed(SEED)
evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT)
NUM_NODES = evaluator.get_num_nodes()
MUT_PROB = 1 / (NUM_NODES * (NUM_NODES+1) / 2)  # expected value is 1 mutation in the matrix

print(SERIES_ADDRESS)
print(ADJ_ADDRESS)
print('CONTINUATION: ' + str(CONTINUATION))
print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
print('SEED: ' + str(SEED))
print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
print('POP_SIZE: ' + str(POP_SIZE))
print('NEWPOP_SIZE: ' + str(NEWPOP_SIZE))
print('NUM_GEN: ' + str(NUM_GEN))
print('CROSS_PROB: ' + str(CROSS_PROB))
print('TEMPERATURE: ' + str(TEMPERATURE))
print('USE_GUIDED_MUTATIONS: ' + str(USE_GUIDED_MUTATIONS))
print('MUT_PROB: ' + str(MUT_PROB))

# load ground truth matrix
with open(ADJ_ADDRESS, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    edges = torch.tensor(edges, dtype=torch.float32)
    gt_matrix = edges.to(torch.float32).to('cpu')
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
tracker = tr.Tracker(gt_matrix, logger)


# for each entry of matrix, flips the bit with probability p. If sym=True, always returns a symmetric matrix
def mutate_full_matrix(matrix, p, sym=False):
    select_mat = (torch.empty((matrix.size()[0], matrix.size()[1])).uniform_() < p).to(matrix.dtype)
    flipped_mat = 1-matrix
    result = select_mat.to(matrix.dtype)*flipped_mat + (1-select_mat).to(matrix.dtype)*matrix
    if sym:
        ut.symmetrize_matrix_(result)
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

def calc_print_save_statistics(population, scores, all_populations, all_scores, mean_scores, trakka):
    # collect statistics for round
    argmax_score = torch.argmax(scores)

    mean_score = scores.mean().item()
    mean_scores.append(mean_score)
    all_scores.append(scores)
    all_populations.append(population)

    # save statistics
    with open(logger.get_path() + '/final_population.pickle', 'wb') as f:
        pickle.dump(population, f)
    with open(logger.get_path() + '/all_populations.pickle', 'wb') as f:
        pickle.dump(all_populations, f)
    with open(logger.get_path() + '/all_scores.pickle', 'wb') as f:
        pickle.dump(all_scores, f)
    with open(logger.get_path() + '/mean_scores.pickle', 'wb') as f:
        pickle.dump(mean_scores, f)

    # print round summary
    print('Mean score: ' + str(mean_score))
    trakka.track(population[argmax_score], losses[argmax_score].item(), scores[argmax_score].item())


def calc_guided_mutation_probs(matrix, sym=True, softmax_factor=1., allow_mut_on_diag=False):
    if sym:
        ut.symmetrize_matrix_(matrix.grad, take_mean=True)
    minus_grad = -1 * matrix.grad
    flipped_matrix = 1 - matrix.detach()
    gradient_partially_flipped = flipped_matrix * minus_grad + matrix.detach() * matrix.grad
    #print(gradient_partially_flipped)
    softmax = torch.nn.Softmax(dim=0)
    if sym:
        triu_selection_mat = torch.ones_like(gradient_partially_flipped).triu(diagonal=int(not allow_mut_on_diag)) == 1
        triu_vec = gradient_partially_flipped[triu_selection_mat]  # get upper triangular matrix as vector
        probs_vec = softmax(triu_vec * softmax_factor)
        probs = torch.zeros_like(gradient_partially_flipped)
        probs[triu_selection_mat] = probs_vec
    else:
        if allow_mut_on_diag==False:
            print('Caution, directed guided mutation currently always allows mutation of diagonal')
        probs = softmax(gradient_partially_flipped.view(NUM_NODES ** 2) / softmax_factor).view(NUM_NODES, NUM_NODES)
    return probs

# use guided mutation. changes exactly one entry of the matrix
def mutate_matrix_guided(matrix, temper, sym=True):
    probs = calc_guided_mutation_probs(matrix, sym=sym, softmax_factor= 1 / temper)
    #print(probs.cpu().numpy())
    probs_vec = probs.view(-1)  # vectorize
    sample = probs_vec.multinomial(num_samples=1)
    result = matrix.detach().clone()
    i = sample // NUM_NODES
    j = sample % NUM_NODES
    result[i, j] = 1 - result[i, j]
    if sym and i != j:
        result[j,i] = 1 - result[j,i]
    result.requires_grad_(True)
    return result


# ----------------------------------------------------------- NOW WITH GUIDED MUTATION AND USING common
print('--------------------- Start training ------------------------')

# initialization
if not CONTINUATION:
    mean_scores = list()
    all_scores = list()
    all_populations = list()
    population_list = []
    for i in range(POP_SIZE):
        random_matrix = ut.sample_undirected_matrix_uniform(NUM_NODES)
        population_list.append(random_matrix if USE_GUIDED_MUTATIONS else random_matrix.requires_grad_(False))
    population = population_list
    scores, losses = evaluator.evaluate_population(population)
else:  # load statistics from previous run
    with open(CONT_ADDRESS + '/final_population.pickle', 'rb') as f:
        population = pickle.load(f)
    with open(CONT_ADDRESS + '/all_populations.pickle', 'rb') as f:
        all_populations = pickle.load(f)
    with open(CONT_ADDRESS + '/all_scores.pickle', 'rb') as f:
        all_scores = pickle.load(f)
    with open(CONT_ADDRESS + '/mean_scores.pickle', 'rb') as f:
        mean_scores = pickle.load(f)
    for mat in population:
        if USE_GUIDED_MUTATIONS:
            mat.requires_grad_(True)
    scores, losses = evaluator.evaluate_population(population)

# evolve for NUM_GEN generations
for j in range(NUM_GEN):
    if not (j==0 and CONTINUATION):
        print('Round ' + str(j) + ' done. -----------------------------------------')
        calc_print_save_statistics(population, scores, all_populations, all_scores, mean_scores, tracker)
        # print mutation probs of best individual
        if USE_GUIDED_MUTATIONS:
            np.set_printoptions(floatmode='fixed', suppress=True, precision=5, linewidth=300)
            print(calc_guided_mutation_probs(population[0], softmax_factor=1 / TEMPERATURE).data.cpu().numpy())

    # select
    idx = torch.randint(high=POP_SIZE, size=(NEWPOP_SIZE,), dtype=torch.long)

    # new generation
    newpop = list()
    for i in range(0, NEWPOP_SIZE, 2):
        # crossover
        if torch.rand(1).item() < CROSS_PROB:
            newind1, newind2 = crossover_onenode(population[idx[i]], population[idx[i+1]])
            newpop.append(newind1)
            newpop.append(newind2)
        else:
            newpop.append(population[idx[i]])
            newpop.append(population[idx[i+1]])
        # mutate
        if USE_GUIDED_MUTATIONS:
            newpop[i] = mutate_matrix_guided(newpop[i], TEMPERATURE)
            newpop[i+1] = mutate_matrix_guided(newpop[i+1], TEMPERATURE)
        else:
            newpop[i] = mutate_full_matrix(newpop[i], MUT_PROB, sym=True)
            newpop[i+1] = mutate_full_matrix(newpop[i+1], MUT_PROB, sym=True)

    # evaluate old and new population and merge
    for ind in population + newpop:
        if ind.grad is not None and USE_GUIDED_MUTATIONS:
            ind.grad.zero_()
    scores, losses = evaluator.evaluate_population(population)
    newscores, newlosses = evaluator.evaluate_population(newpop)
    # update population
    fullpop = population + newpop
    fullscores = torch.cat((scores, newscores))
    fulllosses = torch.cat((losses, newlosses))
    (scores, best_individual_indices) = fullscores.topk(POP_SIZE)
    ind_list = best_individual_indices.tolist()
    population = [fullpop[ind] for ind in ind_list]
    losses = [fulllosses[ind] for ind in ind_list]


calc_print_save_statistics(population, scores, all_populations, all_scores, mean_scores, tracker)