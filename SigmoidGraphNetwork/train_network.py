#from importlib._bootstrap import _installed_safely

import load_data as ld
import torch
import numpy as np
import models as mo
import training_utils as tu
import tracker as tr
import logger as lo
import torch.optim as optim
import sys
import pickle

USE_GPU = True
SERIES_ADDRESS = '../data/final/netrd/SIS/timeseries_ba10_5k_0.15.pickle'
ADJ_ADDRESS = '../data/final/edges_ba10.pickle'
SEED = 0
BATCH_SIZE = 100
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS_PER_CYCLE = 10
NUM_NET_EPOCHS_PER_CYCLE = 20
NUM_CYCLES = 100
FORMAT = 'timeseries'
USE_GUMBEL = True
TEMP_DROP_FACTOR = .95
EXPERIMENTS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if USE_GPU else 'cpu'

orig_terminal = sys.stdout
exp_final_accs = list()
exp_final_tprs = list()
exp_final_fprs = list()
for _ in range(EXPERIMENTS):
    logger = lo.Logger('GGN_logs/final/SIS_ba10_5k' if USE_GUMBEL else 'SGN_logs/EXP_SIS_FIXED_ba10', original_terminal=orig_terminal)
    sys.stdout = logger

    print(SERIES_ADDRESS)
    print(ADJ_ADDRESS)
    print('SEED: ' + str(SEED))
    print('BATCH_SIZE: ' + str(BATCH_SIZE))
    print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
    print('NUM_DYN_EPOCHS_PER_CYCLE: ' + str(NUM_DYN_EPOCHS_PER_CYCLE))
    print('NUM_NET_EPOCHS_PER_CYCLE: ' + str(NUM_NET_EPOCHS_PER_CYCLE))
    print('NUM_CYCLES: ' + str(NUM_CYCLES))
    print('USE_GUMBEL: ' + str(USE_GUMBEL))
    print('EXPERIMENTS: ' + str(EXPERIMENTS))

    # load ground truth matrix
    with open(ADJ_ADDRESS, 'rb') as f:
        edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
        edges = torch.from_numpy(edges)
        gt_matrix = edges.to(torch.float32).to('cpu')
        np.savetxt(logger.path + '/gt_matrix.txt', gt_matrix.numpy().astype(int), fmt='%i')

    tracker = tr.Tracker(gt_matrix, logger)
    data_loader,_, is_continuous, num_nodes = ld.load_data(SERIES_ADDRESS, FORMAT, BATCH_SIZE, False)

    # initialize network
    dyn_learner = mo.GraphNetwork(data_loader.dataset.size()[-1], HIDDEN_SIZE, not is_continuous).to(device)
    network_gen = (mo.GumbelGenerator(num_nodes, temp_drop_frac=TEMP_DROP_FACTOR) if USE_GUMBEL else mo.SigmoidMatrix(num_nodes)).to(device)

    optimizer_dyn = optim.Adam(dyn_learner.parameters(), lr=0.001 if not is_continuous else 0.0001)
    optimizer_net = optim.Adam(network_gen.parameters(), lr=0.1)

    for cycle in range(NUM_CYCLES):
        np.set_printoptions(precision=4, floatmode='fixed', linewidth=1000, suppress=True)
        #print('Raw Matrix: ')
        #print(network_gen.get_matrix(raw=True).detach().cpu().numpy())
        #print('Matrix: ')
        #print(network_gen.get_matrix().detach().cpu().numpy())

        # TRAIN DYNAMICS
        mean_loss = tu.train_dynamics(dyn_learner, network_gen, optimizer_dyn, data_loader, NUM_DYN_EPOCHS_PER_CYCLE, device, is_continuous, cycle==0, tracker)
        # TRAIN NETWORK
        mean_loss = tu.train_network(dyn_learner, network_gen, optimizer_net, data_loader, NUM_NET_EPOCHS_PER_CYCLE, device, is_continuous)
        if USE_GUMBEL:
            network_gen.drop_temperature()
        print('Tracking cycle ' + str(cycle))
        tracker.track(network_gen.get_matrix_hard(), torch.tensor(mean_loss))

    exp_final_accs.append(tracker.get_last_acc())
    exp_final_fprs.append(tracker.get_last_fpr())
    exp_final_tprs.append(tracker.get_last_tpr())

print('----------------- Final results across all experiments ---------------')
print('accs: ' + str(exp_final_accs))
print('tprs: ' + str(exp_final_tprs))
print('fprs: ' + str(exp_final_fprs))
mean_acc = torch.tensor(exp_final_accs).mean().item()
mean_tpr = torch.tensor(exp_final_tprs).mean().item()
mean_fpr = torch.tensor(exp_final_fprs).mean().item()
print('mean acc: ' + str(mean_acc))
print('mean tpr: ' + str(mean_tpr))
print('mean fpr: ' + str(mean_fpr))
