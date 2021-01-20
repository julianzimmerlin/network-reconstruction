import torch.optim as optim
import torch
import models
import load_data as ld
import training_utils as tu
import utils as ut

USE_GPU_DYN = True
DEVICE_DYN = 'cuda' if USE_GPU_DYN else 'cpu'


# this class is used to evaluate the quality of individiual/populations of adjacency matrices for dynamics prediction
# holds data loaders as attributes so that they have to be loaded only once
class Evaluator:
    def __init__(self, series_address, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, USE_OLD_DISCRETE_FORMAT, USE_MAX=False):
        self.NUM_DYN_EPOCHS = NUM_DYN_EPOCHS # if this is -1, it triggers automated convergence detection instead of a fixed number of training epochs
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.USE_MAX = USE_MAX
        self.DETECT_EARLY_CONVERGENCE = DETECT_EARLY_CONVERGENCE
        self.series_address = series_address
        self.data_loader, self.IS_CONTINUOUS, self.NUM_NODES = ld.load_data(series_address, USE_OLD_DISCRETE_FORMAT, BATCH_SIZE)
        self.NUM_BATCHES = len(self.data_loader)
        self.NUM_SAMPLES = self.data_loader.dataset.size()[0]
        print('NUM_SAMPLES: ' + str(self.NUM_SAMPLES))
        print('BATCH_SIZE: ' + str(BATCH_SIZE))
        print('HIDDEN_SIZE: ' + str(HIDDEN_SIZE))
        print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
        print('IS_CONTINUOUS: ' + str(self.IS_CONTINUOUS))
        print('NUM_BATCHES: ' + str(self.NUM_BATCHES))
        print('USE_MAX: ' + str(USE_MAX))

    def evaluate_individual(self, matrix_in,  NUM_DYN_EPOCHS=0, dyn_learner=None, optimizer=None, get_gradient=True):
        if NUM_DYN_EPOCHS <= 0:
            NUM_DYN_EPOCHS = self.NUM_DYN_EPOCHS
        if dyn_learner == None:
            dyn_learner = models.GraphNetwork(self.data_loader.dataset.size()[-1], self.HIDDEN_SIZE, not self.IS_CONTINUOUS)
        if optimizer == None:
            optimizer = optim.Adam(dyn_learner.parameters(), lr=0.001 if not self.IS_CONTINUOUS else 0.0001)
        if USE_GPU_DYN:
            dyn_learner = dyn_learner.cuda()
            matrix = matrix_in.cuda()

        ep = 0
        mean_losses = list()
        while (ep<NUM_DYN_EPOCHS and (not ut.has_converged(mean_losses) if self.DETECT_EARLY_CONVERGENCE else True)):
            losses = []
            for batch_idx, data in enumerate(self.data_loader):
                # data_train: BATCH_SIZE x NUM_NODES x NUM_STEPS x INPUT_SIZE double
                if USE_GPU_DYN:
                    data = data.cuda()
                loss = tu.train_dynamics_learner_batch(optimizer, dyn_learner, matrix.detach(),
                                                 data, DEVICE_DYN, self.IS_CONTINUOUS, optimize=True)
                losses.append(loss)
            mean_loss = torch.stack(losses).mean().cpu()
            mean_losses.append(mean_loss)
            #print('Loss in Epoch ' + str(ep) + ': ' + str(mean_loss.cpu().item()))
            ep += 1

        # one more pass over the training data to calculate loss and possibly gradient on structure
        score, final_loss = self.evaluate_individual_no_training(matrix_in, dyn_learner, get_gradient=get_gradient)

        print('Mean loss in evaluation epoch: ' + str(final_loss) + ', Score: ' + str(score) + ', epochs needed: ' + str(ep))
        return score, final_loss, dyn_learner, optimizer
    
    def evaluate_individual_no_training(self, matrix_in, dyn_learner, get_gradient=True):
        if USE_GPU_DYN:
            dyn_learner = dyn_learner.cuda()
            matrix = matrix_in.cuda()
            
        losses = list()
        for batch_idx, data in enumerate(self.data_loader):
            # data_train: BATCH_SIZE x NUM_NODES x NUM_STEPS x INPUT_SIZE double
            if USE_GPU_DYN:
                data = data.cuda()
            loss = tu.train_dynamics_learner_batch(None, dyn_learner,
                                                   matrix if get_gradient else matrix.detach(),
                                                   data, DEVICE_DYN, self.IS_CONTINUOUS, optimize=False)
            losses.append(loss)
        mean_loss = torch.stack(losses).mean().cpu()

        if get_gradient:
            # normalize gradient -  this is necessary for guided mutation temperature to be independent of data set size
            matrix_in.grad = 1 / self.NUM_BATCHES * matrix_in.grad

        score = torch.exp(-mean_loss).item()  # TODO might not be ideal for continuous dynamics
        return score, mean_loss.item()
    
    # n: number of evaluations
    # use_max: if false: return mean over all evaluations, if True: returns max score and min loss across 
    def eval_individual_n_times(self, mat, n, get_gradient=True):
        scores = list()
        losses = list()
        for i in range(n):
            # mat.grad.zero_()  # not sure if this is good
            score, loss = self.evaluate_individual(mat, get_gradient=get_gradient)
            #print('loss in single run: ' + str(loss))
            scores.append(score)
            losses.append(loss)
        if self.USE_MAX:
            return max(scores), min(losses)
        else:
            return sum(scores) / len(scores), sum(losses) / len(losses)

    def get_num_nodes(self):
        return self.NUM_NODES

    def get_num_samples(self):
        return self.data_loader.dataset.size()[0]

    def get_num_batches(self):
        return len(self.data_loader)

    # evaluates the fitness of all individuals in a population
    def evaluate_population(self, population, num_epochs, dynamics_learners, optimizers, get_gradient):
        scores = torch.zeros(len(population))
        losses = torch.zeros(len(population))
        dyn_learners_out = [None for _ in range(len(population))]
        optimizers_out = [None for _ in range(len(population))]
        for i in range(len(population)):
            print('evaluating individudal ' + str(i) + ' - ' + ut.hash_tensor(population[i]) + '. ', end='')
            scores[i], losses[i], dyn_learners_out[i], optimizers_out[i] = self.evaluate_individual(population[i], NUM_DYN_EPOCHS=num_epochs, dyn_learner=dynamics_learners[i], optimizer=optimizers[i], get_gradient=get_gradient)
        return scores, losses, dyn_learners_out, optimizers_out