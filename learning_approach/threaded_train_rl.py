import os
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from multiprocessing import cpu_count
import argparse
import time


def worker_p(config):
    command = 'python ./learning_approach/train_rl_agent.py'
    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    time.sleep(1)
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-a', type=str, default='ddpg')
    parser.add_argument('-seeds', nargs='+', type=int)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-tau', type=float, default=1e-3)
    parser.add_argument('-explr_p', type=float, default=0.3)
    args = parser.parse_args()

    configs = []
    seeds = range(args.seeds[0], args.seeds[1]) if args.seeds is not None else range(20)
    setup = vars(args)
    for seed in seeds:
        config={}
        for key, val in zip(setup.keys(), setup.values()):
            if key == 'seeds':
                continue
            config[key] = val
        config['seed'] = seed
        configs.append(config)
    n_workers = cpu_count()
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
