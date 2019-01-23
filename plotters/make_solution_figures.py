from problem_environments.conveyor_belt_env import ConveyorBelt
from problem_environments.namo_env import NAMO
from plot_planners import get_result_dir
import argparse
import os
import cPickle as pickle
import numpy as np

parser = argparse.ArgumentParser(description='MCTS parameters')
parser.add_argument('-sampling_strategy', type=str, default='unif')
parser.add_argument('-problem_idx', type=int, default=0)
parser.add_argument('-domain', type=str, default='namo')

args = parser.parse_args()

domain = args.domain
pidx = args.problem_idx


def find_the_best_solution():
    result_dir = get_result_dir(domain, args.sampling_strategy, 0.8, 1.0, 50, 1000)
    max_rwd = -np.inf
    for fin in os.listdir(result_dir):
        if fin.find('pidx') == -1:
            continue
        sd = int(fin.split('_')[2])
        file_pidx = int(fin.split('_')[-1].split('.')[0])

        if file_pidx != pidx:
            continue
        if fin.find('.pkl') == -1:
            continue
        result = pickle.load(open(result_dir + fin, 'r'))

        if domain == 'convbelt':
            file_max_rwd = np.max(np.array(result['search_time'])[:, 2])
        else:
            file_max_rwd = np.max(np.array(result['search_time']['namo'])[:, 2])
        if result['plan'] is not None:
            print len(result['plan'][0]), result['plan'][2]

        #if file_max_rwd > max_rwd:
        #    import pdb;pdb.set_trace()


def main():
    best_plan = find_the_best_solution()
    return
    if domain == 'namo':
        problem_env = NAMO(pidx)
    elif domain == 'convbelt':
        problem_env = ConveyorBelt(pidx)


if __name__ == '__main__':
    main()

