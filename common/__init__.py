import utils
import pickle

init_matrix_address = r'D:\Uni\BA\ColabOutputs\hillclimbing_debug\matrices1.pickle'

with open(init_matrix_address, 'rb') as f:
    matrices = pickle.load(f, encoding='latin1')
init_matrix = matrices[-1]

print(utils.hash_tensor(init_matrix))