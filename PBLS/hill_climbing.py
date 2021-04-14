import evaluator as ev
import sys,tools,time
import pickle
from constants import *
torch.manual_seed(1)

logger = tools.Logger('hill_climbing_logs')
sys.stdout = logger
path = logger.get_path()

# initialize evaluator with given timeseries data
series_address = './data/Voter/timeseries_ba20_100.pickle'
evaluator = ev.Evaluator(series_address=series_address, continuous=CONTINUOUS)

# load ground truth matrix
adj_address = './data/Voter/edges_ba20.pickle'

with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)
    np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')

def calc_save_print_statistics(tprs, fprs, scores, cands, cand, score, gen):
    (tpr, fpr) = tools.calculate_tpr_fpr(gt_matrix, cand, skip_diag=True)
    tprs.append(tpr)
    fprs.append(fpr)
    scores.append(score)
    cands.append(cand)
    with open(path + '/scores.pickle', 'wb') as f:
        pickle.dump(scores, f)
    with open(path + '/cands.pickle', 'wb') as f:
        pickle.dump(cands, f)
    with open(path + '/tprs.pickle', 'wb') as f:
        pickle.dump(tprs, f)
    with open(path + '/fprs.pickle', 'wb') as f:
        pickle.dump(fprs, f)
    print(str(gen) + ': Score ' + str(score))
    print('Matrix: \n' + str(cand.data.cpu().numpy().astype(int)))
    print('TPR: ' + str(tpr))
    print('FPR: ' + str(fpr))

def eval_n_times(n, mat, continuous=False):
    scores = list()
    for i in range(n):
        # mat.grad.zero_()  # not sure if this is good
        scores.append(evaluator.evaluate_individual_by_loss(mat, continuous))
    return sum(scores) / len(scores)
# -----------------------------------------------------------------------
tprs = list()
fprs = list()
scores = list()
cands = list()

start_time = time.process_time()

cand = tools.sample_undirected_matrix_uniform(NUM_NODES)
#cand = torch.zeros(NUM_NODES, NUM_NODES, requires_grad=True)
#cand = gt_matrix.detach().clone().requires_grad_(True)
score = eval_n_times(10, cand, continuous=CONTINUOUS)  # evaluator.evaluate_individual_by_loss(cand)
for gen in range(100):
    calc_save_print_statistics(tprs, fprs, scores, cands, cand, score, gen)

    probs = tools.calc_guided_mutation_probs(cand, softmax_factor=1)
    _, indices = probs.view(-1).topk(10)      # NUM_NODES*(NUM_NODES+1) // 2)
    found_better = False
    new_cands = list()
    for index in indices:
        i = index // NUM_NODES
        j = index % NUM_NODES
        new_cand = cand.detach().clone()
        new_cand[i,j] = 1 - new_cand[i,j]
        if i != j:
            new_cand[j,i] = 1 - new_cand[j,i]
        new_cand.requires_grad_(True)
        new_cands.append(new_cand)
    for new_cand in new_cands:
        tpr, fpr = tools.calculate_tpr_fpr(gt_matrix, new_cand, skip_diag=True)
        print('Trying out candidate with TPR ' + str(tpr) + ', FPR ' + str(fpr))
        new_score = eval_n_times(10, new_cand, continuous=CONTINUOUS)#evaluator.evaluate_individual_by_loss(new_cand)
        if new_score > score:
            cand = new_cand
            score = new_score
            found_better = True
            break
    if not found_better:
        print('Maybe a local minimum. Repeating with 10 evaluations to make sure.')
        score = eval_n_times(100, cand)
        print('Current AVG score to beat: ' + str(score))
        for new_cand in new_cands:
            new_score = eval_n_times(30, new_cand, continuous=CONTINUOUS)
            print('Candidate AVG score:' + str(new_score))
            if new_score > score:
                cand = new_cand
                score = new_score
                found_better = True
                break
        if not found_better:
            print('It seems to really be a local minimum')
            break


calc_save_print_statistics(tprs, fprs, scores, cands, cand, score, gen+1)
print('Time needed: ' + str(time.process_time() - start_time) + ' sec')