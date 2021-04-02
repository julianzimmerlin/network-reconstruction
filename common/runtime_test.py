import pickle
import torch
import numpy as np
import models
import time

SEED = 1
HIDDEN_SIZE = 128
BATCH_SIZE =10
IS_CONTINUOUS = True
INPUT_SIZE = 1

print('SEED: {}'.format(SEED))
print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
torch.manual_seed(SEED)
np.random.seed(SEED)

matrix_sizes = [a*10 for a in list(range(1,55))]
dyn_learner = models.GraphNetwork(INPUT_SIZE, HIDDEN_SIZE, IS_CONTINUOUS)
dyn_learner = dyn_learner#.cuda()
times = list()
for i, sz in enumerate(matrix_sizes):
    #mat = torch.randint(0,2,(sz,sz)).to(torch.float32)#.cuda()
    mat = torch.zeros((sz, sz)).to(torch.float32)
    features = torch.randn((BATCH_SIZE, sz, INPUT_SIZE))#.cuda()
    #print(features)

    adjs = mat.repeat(BATCH_SIZE, 1, 1)

    start_time = time.process_time()
    output = dyn_learner(features, adjs)
    times.append(time.process_time() - start_time)
    print("Size " + str(sz) + " time: " + str(times[i]))
print(times)
