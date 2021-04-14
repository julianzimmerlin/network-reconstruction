import evaluator as ev
import sys,tools
import pickle
from constants import *

logger = tools.Logger('weight_similarity_logs')
sys.stdout = logger
path = logger.get_path()

# initialize evaluator with given timeseries data
series_address = './data/Voter/timeseries_Voter_grid5x2.pickle'
evaluator = ev.Evaluator(series_address=series_address)

# load ground truth matrix
adj_address = './data/VOTER/edges_Voter_grid5x2.pickle'
with open(adj_address, 'rb') as f:
    edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
    gt_matrix = torch.tensor(edges, dtype=torch.float32)

for _ in range(10):
    evaluator.evaluate_individual_by_loss(gt_matrix)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
mut_matrix = gt_matrix.detach().clone()
mut_matrix[0,2] = mut_matrix[2,0] = 1

for _ in range(10):
    evaluator.evaluate_individual_by_loss(mut_matrix)