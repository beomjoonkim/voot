import os
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
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
    n_workers = int(3)
    print configs
    pool = ThreadPool(n_workers)
    import pdb;pdb.set_trace()
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
