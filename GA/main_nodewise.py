import sys, pickle
import torch
import evaluator as ev
import logger as lo
import numpy as np
import tracker as tr
import utils as ut
import search_utils as su

SEED =0
SERIES_ADDRESS = r'../data/final/Voter/timeseries_ba20_100.pickle'
ADJ_ADDRESS = r'../data/final/edges_ba20.pickle'
BATCH_SIZE = 100
HIDDEN_SIZE = 128
NUM_DYN_EPOCHS_INIT = 300
NUM_DYN_EPOCHS = 300
DETECT_EARLY_CONVERGENCE = False
RESET_DYN_LEARNER_EVERY_NTH_GEN = 1
POP_SIZE = 8
NEWPOP_SIZE = 16
NUM_GEN = 45
USE_NODEWISE_EVALUATION = False
USE_EVALEPOCH_FOR_GUIDED_MUTATION = True
USE_DYNAMIC_MUTATIONS = False
USE_DETERMINISTIC_EVAL = False
FORMAT = 'old'
EXPERIMENTS = 5

CONTINUATION = False
CONT_ADDRESS = r'D:\Uni\BA\ColabOutputs\ba20\2020-12-10T22_50_22.113418'
torch.manual_seed(SEED)
np.random.seed(SEED)


def calc_print_save_statistics(population, all_populations, losses, trakka):
    # collect statistics for round
    if USE_NODEWISE_EVALUATION:
        losses = losses.mean(dim=1)
    argmin_loss = torch.argmin(losses)

    all_populations.append(population)
    # save statistics
    with open(logger.get_path() + '/final_population.pickle', 'wb') as f:
        pickle.dump(population, f)
    with open(logger.get_path() + '/all_populations.pickle', 'wb') as f:
        pickle.dump(all_populations, f)
    trakka.track(population[argmin_loss], losses[argmin_loss])

# ----------------------------------------------------------- EARLY TERMINATION, CONTINUOUS TRAINING

orig_terminal = sys.stdout
exp_final_accs = list()
exp_final_tprs = list()
exp_final_fprs = list()
for _ in range(EXPERIMENTS):
    logger = lo.Logger('GA_logs/final/Voter_ba10_100_eval_doublemut')
    sys.stdout = logger

    evaluator = ev.Evaluator(SERIES_ADDRESS, NUM_DYN_EPOCHS, DETECT_EARLY_CONVERGENCE, BATCH_SIZE, HIDDEN_SIZE, FORMAT, not USE_EVALEPOCH_FOR_GUIDED_MUTATION, USE_NODEWISE_EVALUATION, DETERMINISTIC=USE_DETERMINISTIC_EVAL)
    NUM_NODES = evaluator.get_num_nodes()

    print(SERIES_ADDRESS)
    print(ADJ_ADDRESS)
    print('CONTINUATION: ' + str(CONTINUATION))
    print('CONT_ADDRESS: ' + str(CONT_ADDRESS))
    print('SEED: ' + str(SEED))
    print('NUM_DYN_EPOCHS: ' + str(NUM_DYN_EPOCHS))
    print('NUM_DYN_EPOCHS_INIT: ' + str(NUM_DYN_EPOCHS_INIT))
    print('DETECT_EARLY_CONVERGENCE: ' + str(DETECT_EARLY_CONVERGENCE))
    print('RESET_DYN_LEARNER_EVERY_NTH_GEN: ' + str(RESET_DYN_LEARNER_EVERY_NTH_GEN))
    print('POP_SIZE: ' + str(POP_SIZE))
    print('NEWPOP_SIZE: ' + str(NEWPOP_SIZE))
    print('NUM_GEN: ' + str(NUM_GEN))
    print('USE_NODEWISE_EVALUATION: ' + str(USE_NODEWISE_EVALUATION))
    print('USE_EVALEPOCH_FOR_GUIDED_MUTATION: ' + str(USE_EVALEPOCH_FOR_GUIDED_MUTATION))
    print('USE_DETERMINISTIC_EVAL: ' + str(USE_DETERMINISTIC_EVAL))

    # load ground truth matrix
    with open(ADJ_ADDRESS, 'rb') as f:
        edges = pickle.load(f, encoding='latin1')  # (num_nodes x num_nodes) 1 or 0
        edges = torch.tensor(edges, dtype=torch.float32)
        gt_matrix = edges.to(torch.float32).to('cpu')
        np.savetxt(logger.get_path() + '/ground_truth_matrix.txt', gt_matrix.numpy(), fmt='%i')
    tracker = tr.Tracker(gt_matrix, logger)

    print('--------------------- Start training ------------------------')

    # initialization
    if not CONTINUATION:
        all_populations = list()
        population_list = []
        for i in range(POP_SIZE):
            random_matrix = ut.sample_undirected_matrix_uniform(NUM_NODES, requires_grad=not USE_EVALEPOCH_FOR_GUIDED_MUTATION)
            population_list.append(random_matrix)
        population = population_list
        dynamics_learners = [None for _ in range(POP_SIZE)]
        optimizers = [None for _ in range(POP_SIZE)]
    else:  # load statistics from previous run
        with open(CONT_ADDRESS + '/final_population.pickle', 'rb') as f:
            population = pickle.load(f)
        with open(CONT_ADDRESS + '/all_populations.pickle', 'rb') as f:
            all_populations = pickle.load(f)
        with open(CONT_ADDRESS + '/dyn_learners.pickle', 'rb') as f:
            dynamics_learners = pickle.load(f)
        with open(CONT_ADDRESS + '/optimizers.pickle', 'rb') as f:
            optimizers = pickle.load(f)
        for mat in population:
            mat.requires_grad_(not USE_EVALEPOCH_FOR_GUIDED_MUTATION)
    losses, dynamics_learners, optimizers = evaluator.evaluate_population(population, NUM_DYN_EPOCHS_INIT, dynamics_learners, optimizers)

    # evolve for NUM_GEN generations
    for j in range(NUM_GEN):
        if not (j==0 and CONTINUATION):
            print('Round ' + str(j) + ' done. -----------------------------------------')
            calc_print_save_statistics(population, all_populations, losses, tracker)
            with open(logger.get_path() + '/dyn_learners.pickle', 'wb') as f:
                pickle.dump(dynamics_learners, f)
            with open(logger.get_path() + '/optimizers.pickle', 'wb') as f:
                pickle.dump(optimizers, f)

        with open(logger.get_path() + '/lineage.txt', "a") as f:
            if j > 0:
                f.write('Parents: \n')
                for par in parentids:
                    f.write(str(par.to(torch.int).item()) + '. ')
                f.write('\n')
            f.write("Generation " + str(j) + ': \n')
            for indiv in population:
                f.write(ut.hash_tensor(indiv) + '. ')
            f.write('\n')

        # select
        idx = torch.randint(high=POP_SIZE, size=(NEWPOP_SIZE,), dtype=torch.long)

        # new generation
        newpop = list()
        for i in range(NEWPOP_SIZE):
            # mutate
            if USE_DYNAMIC_MUTATIONS:
                next_indiv,changed_indices = ut.exec_dynamic_step_eval(population[idx[i]], dynamics_learners[idx[i]], evaluator, losses[idx[i]])\
                    if USE_EVALEPOCH_FOR_GUIDED_MUTATION else ut.exec_dynamic_step_grad(population[idx[i]])
                newpop.append(next_indiv.requires_grad_(True))
            else:
                newpop.append(su.double_mutation(population[idx[i]], dynamics_learners[idx[i]], evaluator))

        # evaluate old and new population and merge
        [ind.grad.zero_() if ind.grad is not None else None for ind in population + newpop]
        if j % RESET_DYN_LEARNER_EVERY_NTH_GEN == 0: # reset dyn_learners every n-th generation
            dynamics_learners = [None]*len(dynamics_learners)
            optimizers = [None]*len(optimizers)
        newlosses, newdynamics_learners, newoptimizers = evaluator.evaluate_population(newpop, NUM_DYN_EPOCHS_INIT if j % RESET_DYN_LEARNER_EVERY_NTH_GEN == 0 else NUM_DYN_EPOCHS,
                                                                                [dynamics_learners[id] for id in idx], [optimizers[id] for id in idx])
        if not USE_DETERMINISTIC_EVAL:
            losses, dynamics_learners, optimizers = evaluator.evaluate_population(population, NUM_DYN_EPOCHS_INIT if j % RESET_DYN_LEARNER_EVERY_NTH_GEN == 0 else NUM_DYN_EPOCHS,
                                                                                dynamics_learners, optimizers)
        # update population
        fullpop = population + newpop
        fulllosses = torch.cat((losses, newlosses))
        fulldyn_learners = dynamics_learners + newdynamics_learners
        fulloptimizers = optimizers + newoptimizers
        fullparentids = torch.cat((torch.arange(0,len(population)), idx))

        if USE_NODEWISE_EVALUATION:
            (_, sorted_indices) = su.sort_matrix_set_relevant_mutations(fullpop, fulllosses)
        else:
            (_, sorted_indices) = fulllosses.topk(POP_SIZE, largest=False)
        ind_list = sorted_indices[:POP_SIZE].tolist()

        population = [fullpop[ind] for ind in ind_list]
        losses = fulllosses[ind_list]
        dynamics_learners = [fulldyn_learners[ind] for ind in ind_list]
        optimizers = [fulloptimizers[ind] for ind in ind_list]
        parentids = [fullparentids[ind] for ind in ind_list]

    calc_print_save_statistics(population, all_populations, losses, tracker)

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
