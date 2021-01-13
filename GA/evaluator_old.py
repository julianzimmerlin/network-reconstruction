import tools
import torch.optim as optim
import models
import time
from constants import *


# this class is used to evaluate the quality of individiual/populations of adjacency matrices for dynamics prediction
# holds data loaders as attributes so that they have to be loaded only once
class Evaluator:

    def __init__(self, series_address, continuous=False):
        self.series_address = series_address
        if continuous:
            self.data_loader = tools.load_data_cont(series_address, batch_size=BATCH_SIZE)
        else:
            self.data_loader = tools.load_data(series_address, batch_size=BATCH_SIZE, split_train_test=False)
        print('train set batch num : ' + str(len(self.data_loader)))
        print('BATCH_SIZE:' + str(BATCH_SIZE))
        print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
        print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))

        self.loss_fn = torch.nn.NLLLoss()

    # evaluates the fitness of a population
    def evaluate_pop(self, population, use_accu=False):
        scores = torch.zeros(len(population))

        for i in range(len(population)):
            print('evaluating individudal ' + str(i) + '. ', end='')
            scores[i] = self.evaluate_individual_testset(population[i]) if use_accu \
                else self.evaluate_individual_by_loss(population[i])
        return scores

    # evaluates the fitness of an adjacency matrix after training. The fitness is the mean accuracy on a test set
    def evaluate_individual_by_loss(self, matrix, continuous=False):
        if not continuous:
            self.data_loader = tools.load_data(self.series_address, batch_size=BATCH_SIZE, split_train_test=False)

        # initialize dynamics learner and optimizer
        if continuous:
            dyn_learner = models.GumbelGraphNetworkClf(1)
        else:
            dyn_learner = models.GumbelGraphNetworkClf(2) # model parameters are initialized randomly -> function might return different values in repeated runs

        if USE_GPU_DYN:
            dyn_learner = dyn_learner.cuda()
            matrix = matrix.cuda()
        optimizer = optim.Adam(dyn_learner.parameters(), lr=0.001 if not continuous else 0.0001)

        # train for NUM_DYN_EPOCHS epochs
        for ep in range(NUM_DYN_EPOCHS):
            #print(ep)
            last_ep = (ep == NUM_DYN_EPOCHS-1)
            losses = []

            #start_time = time.process_time()
            for batch_idx, data in enumerate(self.data_loader):
                # data_train: BATCH_SIZE x NUM_NODES x 2 double, data_target: BATCH_SIZE x NUM_Nodes int64
                if USE_GPU_DYN:
                    if continuous:
                        data = data.cuda()
                    else:
                        data_train = data[0].cuda()
                        data_target = data[1].cuda()
                if continuous:
                    #start_time = time.process_time()
                    loss = self.train_dynamics_learner(optimizer, dyn_learner, matrix if last_ep else matrix.detach(), data, 10, optimize=not last_ep)
                    #print('Train time: ' + str(time.process_time() - start_time))
                else:
                    loss = self.train_batch_dyn(optimizer,dyn_learner,matrix if last_ep else matrix.detach(),
                                            data_train,data_target,self.loss_fn, optimize=not last_ep)
                losses.append(loss)
            #print('Loss in Epoch ' + str(ep) + ': ' + str(torch.stack(losses).mean().cpu().item()))
        #print('Network parameters:')
        #print(dyn_learner.edge1.weight)
        mean_loss = torch.stack(losses).mean().cpu()
        score = torch.exp(-mean_loss).item()
        #print('Mean loss in last epoch: ' + str(mean_loss.item()) + ', Score: ' + str(score))
        return score

    # evaluates the fitness of an adjacency matrix after training. fitness = e^-mean(loss) in last epoch
    def evaluate_individual_testset(self, matrix):
        self.train_data_loader, self.test_data_loader = tools.load_data(self.series_address, batch_size=BATCH_SIZE, split_train_test=True)

        # initialize dynamics learner and optimizer
        dyn_learner = models.GumbelGraphNetworkClf(2, HIDDEN_SIZE)  # model parameters are initialized randomly -> function might return different values in repeated runs

        if USE_GPU_DYN:
            dyn_learner = dyn_learner.cuda()
            matrix = matrix.cuda()
        optimizer = optim.Adam(dyn_learner.parameters(), lr=0.001)

        # train for NUM_DYN_EPOCHS epochs
        for ep in range(NUM_DYN_EPOCHS):
            for batch_idx, (data_train, data_target) in enumerate(self.train_data_loader):
                # data_train: BATCH_SIZE x NUM_NODES x 2 double, data_target: BATCH_SIZE x NUM_Nodes int64
                if USE_GPU_DYN:
                    data_train = data_train.cuda()
                    data_target = data_target.cuda()
                loss = self.train_batch_dyn(optimizer, dyn_learner, matrix, data_train, data_target, self.loss_fn)
                if batch_idx == 0:
                    # firstacc = accu
                    firstloss = loss

        # test the dynamics learner
        accu_all = []
        for batch_idx, (data_test, data_target) in enumerate(self.test_data_loader):
            if USE_GPU_DYN:
                data_test = data_test.cuda()  # TODO why double?
                data_target = data_target.cuda()
            accu = self.evaluate_test_accuracy(matrix, dyn_learner, data_test, data_target)
            accu_all.append(accu)
        avg_accuracy = torch.mean(torch.tensor(accu_all)).item()
        print('Accuracy on test set: ' + str(avg_accuracy))
        return avg_accuracy

    # train dyn trainer
    def train_batch_dyn(self, optimizer, dyn_learner, adj, data_train, data_target, loss_fn, optimize=True):

        optimizer.zero_grad()
        adj = adj.repeat(data_train.size()[0], 1, 1)  # copy it BATCH_SIZE times to form a BATCH_SIZE x NUM_NODES x NUM_NODES tensor
        output = dyn_learner(data_train, adj)  # 128x10x2 - 2 for encoded states? gives floats
        output = output.permute(0, 2, 1)  # 128x2x10 - required format for NLLLoss
        data_target = data_target.long()
        # data_target = data_target.long()
        # accus = cacu_accu(output, data_target)
        loss = loss_fn(output, data_target)
        loss.backward()

        # optimizer for dyn learner
        if optimize:
            optimizer.step()

        return loss#, accus

    def train_dynamics_learner(self, optimizer, dynamics_learner, relations, data, steps, optimize=True):
        # dynamics_learner.train()

        optimizer.zero_grad()


        adjs = relations.unsqueeze(0)
        adjs = adjs.repeat(data.size()[0], 1, 1)
        adjs = adjs.cuda() if USE_GPU_DYN else adjs

        input = data[:, :, 0, :]
        target = data[:, :, 1: steps, :]
        output = input

        outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
        outputs = outputs.cuda() if USE_GPU_DYN else outputs
        # 完成steps-1步预测，output格式为：batchsize, num_nodes, time_steps, dimension
        for t in range(steps - 1):
            output = dynamics_learner(output, adjs)
            outputs[:, :, t, :] = output


        loss = torch.mean(torch.abs(outputs - target))
        loss.backward()
        if optimize:
            optimizer.step()
        #mse = F.mse_loss(outputs, target)
        #torch.cuda.synchronize()
        #start_time = time.process_time()
        if USE_GPU_DYN:
            loss = loss.cpu()
        #    mse = mse.cpu()
        #torch.cuda.synchronize()
        #print('Adj cuda time: ' + str(time.process_time() - start_time))

        return loss#, mse

    # returns: accuracy ([0,1] standard python double) of predictions on test batch
    def evaluate_test_accuracy(self, adjmat, dyn_learner, data_train, data_target):
        # get predictions from dyn_learner
        adjmat = adjmat.repeat(data_train.size()[0], 1, 1)
        output = dyn_learner(data_train, adjmat)  # output: BATCH_SIZE x NUM_NODES x 2
        # -> just take second feature map for comparison? NO! these are floats (posterior distrib.) but is accu good then?

        # calculate the prediction accuracy
        # data_target = data_target.long() # should be unnecessary
        accu = self.calculate_accuracy(output, data_target)
        return accu

    # calculates accuracy of predictions (by taking argmax of predicted class scores)
    def calculate_accuracy(self, output, target):
        output = output.cpu()  # TODO is this necessary?
        target = target.cpu()
        right = 0.
        for i in range(output.size(0)):
            for j in range(output.size(1)):
                if (output[i][j][0] > output[i][j][1] and target[i][j] == 0) \
                        or (output[i][j][0] < output[i][j][1] and target[i][j] == 1):
                    right += 1
        return right / target.numel()