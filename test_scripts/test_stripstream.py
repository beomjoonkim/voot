from planners.convbelt_stripstream.stripstream import solve_pddlstream as solve_convbelt
from planners.namo_stripstream.stripstream import solve_pddlstream as solve_namo

import os
import pickle
import argparse

import numpy as np
import random

def make_save_dir(domain):
    save_dir = './test_results/' + domain + '_results/stripstream/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return save_dir


def main():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-problem_idx', type=int, default=0)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-v', action='store_true', default=False)
    args = parser.parse_args()
    sd = 0
    np.random.seed(sd)
    random.seed(sd)

    save_dir = make_save_dir(args.domain)
    if args.domain == 'convbelt':
        plan, search_time = solve_convbelt()
    elif args.domain == 'namo':
        plan, search_time = solve_namo()
    else:
        print "Wront domain name"
        return

    pickle.dump({'search_time': search_time, 'plan': plan},
                open(save_dir + str(args.problem_idx) + '.pkl', 'wb'))


if __name__ == '__main__':
    main()
