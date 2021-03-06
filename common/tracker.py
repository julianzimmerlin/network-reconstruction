import utils
import time
import pickle


# this class tracks all the relevant statistics for an experiment
class Tracker:
    def __init__(self, gt_matrix, logger):
        self.gen_num = 0
        self.logger = logger
        self.gt_matrix = gt_matrix
        self.start_time = None
        self.tprs = list()
        self.fprs = list()
        self.accuracies = list()
        #self.scores = list()
        self.matrices = list()
        self.losses = list()
        self.timestamps = list()
        self.hashes = list()

    def track(self, mat, loss, score=0):
        matrix = mat.cpu()
        tpr, fpr = utils.calculate_tpr_fpr(self.gt_matrix, matrix)
        acc = utils.calculate_accuracy(self.gt_matrix, matrix)
        hash = utils.hash_tensor(matrix)

        if self.start_time is None:
            timestamp = 0
            self.start_time = time.process_time()
        else:
            timestamp = time.process_time() - self.start_time

        self.accuracies.append(acc)
        self.tprs.append(tpr)
        self.fprs.append(fpr)
        #self.scores.append(score)
        self.matrices.append(matrix)
        self.losses.append(loss)
        self.timestamps.append(timestamp)
        num_times_seen_before = sum([1 if hash == h else 0 for h in self.hashes])
        self.hashes.append(hash)

        print('------------------------- TRACKING ---------------------------')
        #print('Matrix generation ' + str(self.gen_num) + ':')
        print('Matrix Hash: ' + hash + ', num_times_seen_before: ' + str(num_times_seen_before))
        #print(matrix.data.cpu().numpy().astype(int))
        print('ACC: ' + str(acc))
        print('TPR/FPR: ' + str(tpr) + ', ' + str(fpr))
        print('Loss: ' + str(loss.tolist()))
        #print('Score: '+ str(score))
        print('Timestamp: ' + str(timestamp))
        print('---------------------------------------------------------------')
        self.logger.flush()

        #with open(self.logger.get_path() + '/scores.pickle', 'wb') as f:
        #    pickle.dump(self.scores, f)
        with open(self.logger.get_path() + '/matrices.pickle', 'wb') as f:
            pickle.dump(self.matrices, f)
        with open(self.logger.get_path() + '/tprs.pickle', 'wb') as f:
            pickle.dump(self.tprs, f)
        with open(self.logger.get_path() + '/fprs.pickle', 'wb') as f:
            pickle.dump(self.fprs, f)
        with open(self.logger.get_path() + '/accs.pickle', 'wb') as f:
            pickle.dump(self.accuracies, f)
        with open(self.logger.get_path() + '/losses.pickle', 'wb') as f:
            pickle.dump(self.losses, f)
        with open(self.logger.get_path() + '/timestamps.pickle', 'wb') as f:
            pickle.dump(self.timestamps, f)

        self.gen_num = self.gen_num + 1
    
    def get_last_tpr(self):
        return self.tprs[-1]
    
    def get_last_fpr(self):
        return self.fprs[-1]
    
    def get_last_acc(self):
        return self.accuracies[-1]