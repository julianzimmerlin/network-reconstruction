import sys
import torch
import numpy as np
import pickle
import time
import logger as lo
import evaluator as ev
import utils as ut
import tracker as tr
import copy
import search_utils as su

SEED = 0
SERIES_ADDRESS = r'../data/final/Voter/timeseries_ba300_1k.pickle'
ADJ_ADDRESS = r'../data/final/edges_ba300.pickle'
BATCH_SIZE = 100
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS = 30
DETECT_EARLY_CONVERGENCE = False
FORMAT = 'old'
USE_EVALEPOCH_FOR_GUIDED_MUTATION = False
USE_NODEWISE_LOSS = True
USE_DYNAMIC_STEPS = True
NUM_GEN = 45
DETERMINISTIC_EVAL = False
RANDOM = False
FREE_WALK = False
EXPERIMENTS = 5

CONTINUATION = False
CONT_ADDRESS = '../GA/GA_logs/final/cml_ba20_1k_4_restart_shortcut_eval/2021-03-11T03_17_31.151794'

torch.manual_seed(SEED)
np.random.seed(SEED)

orig_terminal = sys.stdout
exp_final_accs = list()
exp_final_tprs = list()
exp_final_fprs = list()
for _ in range(EXPERIMENTS):
    logger = lo.Logger('hillclimbing_logs/linear/final/nodewise/Voter/ba300_1k', original_terminal=orig_terminal)
    sys.stdout = logger
    print(SERIES_ADDRESS)
    print(ADJ_ADDRESS)
    print('CONTINUATION: ' + str(CONTINUATION))
    print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
    print('SEED: ' + str(SEED))
    print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
    print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
    print('USE_EVALEPOCH_FOR_GUIDED_MUTATION: ' + str(USE_EVALEPOCH_FOR_GUIDED_MUTATION))
    print('USE_NODEWISE_LOSS: ' + str(USE_NODEWISE_LOSS))
    print('USE_DYNAMIC_STEPS: ' + str(USE_DYNAMIC_STEPS))
    print('NUM_GEN: ' + str(NUM_GEN))
    print('DETERMINISTIC_EVAL: ' + str(DETERMINISTIC_EVAL))
    print('FREE WALK: ' + str(FREE_WALK))
    print('EXPERIMENTS: ' + str(EXPERIMENTS))
    print('FORMAT: ' + FORMAT)
    if RANDOM:
        print('ATTENTIONNNNNNNNNNNNNNNNNNNN RANDOM MUTATIONS')

    # initialize evaluator with given timeseries data
    evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, FORMAT, not USE_EVALEPOCH_FOR_GUIDED_MUTATION, USE_NODEWISE_LOSS, DETERMINISTIC=DETERMINISTIC_EVAL)
    NUM_NODES = evaluator.get_num_nodes()
    # load ground truth matrix
    with open(ADJ_ADDRESS, 'rb') as f:
        edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
        edges = torch.tensor(edges, dtype=torch.float32)
        gt_matrix = edges.to(torch.float32).to('cpu')
        np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
    tracker = tr.Tracker(gt_matrix, logger)

    # -----------------------------------------------------------------------


    start_time = time.process_time()
    if CONTINUATION:
        #with open(CONT_ADDRESS+'/matrices.pickle', 'rb') as f:
        #    matrices = pickle.load(f)
        #    cand = matrices[-1]
        with open(CONT_ADDRESS+'/all_populations.pickle', 'rb') as f:
            all_pops = pickle.load(f)
            start_pop = all_pops[7]
            cand = start_pop[5]
    else:
        cand = ut.sample_undirected_matrix_uniform(NUM_NODES)
    #DEBUG::::::::::::
    #cand = gt_matrix.clone().detach()
    #if not USE_EVALEPOCH_FOR_GUIDED_MUTATION:
    #    cand.requires_grad_(True)

    if DETERMINISTIC_EVAL:
        loss,dyn_learner,_ = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, None, None)

    for gen in range(NUM_GEN):
        print('\nGeneration ' + str(gen))
        if not DETERMINISTIC_EVAL:
            loss, dyn_learner, _ = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, None, None)
        tracker.track(cand, loss)

        #indices = ut.calc_mutation_order_evalepoch(cand, dyn_learner, evaluator) if USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.calc_mutation_order_gradient(cand)
        #indices = indices[:MAX_CHANGES] # do MAX_CHANGES most promising mutations
        #print(indices)
        #new_cand = cand.detach().clone()
        #for index in indices:
        #    new_cand[list(index)] = 1 - new_cand[list(index)]
        #    new_cand[list(index.flip(dims=(0,)))] = 1 - new_cand[list(index.flip(dims=(0,)))]
        #new_cand.requires_grad_(not USE_EVALEPOCH_FOR_GUIDED_MUTATION)

        #loss,_,_ = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, copy.deepcopy(dyn_learner), copy.deepcopy(optimizer))
        #print('Reevaluating current cand. ', end='')
        #loss, dyn_learner, optimizer = evaluator.evaluate_individual(cand, NUM_DYN_EPOCHS, None, None)


        if RANDOM:
            new_cand, indices = ut.exec_dynamic_step_random(cand) if USE_DYNAMIC_STEPS else ut.exec_single_step_random(cand)
        else:
            if USE_DYNAMIC_STEPS:
                new_cand, indices = ut.exec_dynamic_step_eval(cand, dyn_learner, evaluator, loss) if USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.exec_dynamic_step_grad(cand)
            else:
                new_cand, indices = ut.exec_single_step_eval(cand, dyn_learner, evaluator, loss) if USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.exec_single_step_grad(cand)

        print(indices)
        if len(indices)==0:
            print('skipping.')
            continue
            #new_cand, indices = ut.exec_dynamic_step_grad(cand)

        new_cand.requires_grad_(True)
        new_loss, new_dyn_learner, _ = evaluator.evaluate_individual(new_cand,NUM_DYN_EPOCHS, None, None)

        if USE_NODEWISE_LOSS:
            count_changes = 0
            cand = cand.detach().clone()
            for index in indices: # check for each mutation if it was good
                if su.compare_matrices_specific_mutation(new_cand, cand, new_loss, loss, index) < 0:
                    count_changes += 1
                    cand[list(index)] = new_cand.detach()[list(index)]
                    cand[list(index.flip(dims=(0,)))] = new_cand.detach()[list(index.flip(dims=(0,)))]
            cand.requires_grad_(True)
            print('count_changes: ' + str(count_changes ) + ' / ' + str(len(indices)))
            #if count_changes == 0:
            #    if MAX_CHANGES > 1:
            #        MAX_CHANGES = MAX_CHANGES // 2
            #    else:
            #        MAX_CHANGES = NUM_NODES
        else:
            #print('new_loss: ' + str(new_loss.item()))
            #print('loss: ' + str(loss.item()))
            if FREE_WALK or new_loss < loss:
                print('Accepting!')
                cand = new_cand
                dyn_learner = new_dyn_learner
                loss = new_loss
            else:
                print('Rejecting!')
    tracker.track(cand, loss)
    print('Time needed: ' + str(time.process_time() - start_time) + ' sec')

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
