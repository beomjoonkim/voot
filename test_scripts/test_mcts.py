from problem_environments.conveyor_belt_env import ConveyorBelt
from problem_environments.namo_env import NAMO
from problem_environments.mover_env import Mover

from planners.mcts import MCTS
from planners.voronoi_mcts import VoronoiMCTS
from generators.PlaceUniform import PlaceUnif
from generators.PickUniform import PickWithoutBaseUnif
from generators.PickUniform import PickWithBaseUnif

from sampling_strategies.voo import VOO, MoverVOO
from sampling_strategies.uniform import Uniform



import argparse
import cPickle as pickle
import os
import openravepy
import numpy as np
import random


def make_save_dir(domain, uct_parameter, widening_parameter, sampling_strategy):
    save_dir = './test_results/' + domain + '_results/uct_' + str(uct_parameter) + '_widening_' \
               + str(widening_parameter) + '_'+sampling_strategy+'/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return save_dir


def make_sampling_strategy(sampling_strategy, domain_name, problem_env):
    if domain_name == 'namo':
        pick_pi = PickWithBaseUnif(problem_env, problem_env.problem['env'].GetRobots()[0],
                                   problem_env.problem['all_region'])
        place_pi = PlaceUnif(problem_env.problem['env'], problem_env.problem['env'].GetRobots()[0],
                             problem_env.obj_region, problem_env.robot_region)
        if sampling_strategy == 'voo':
            sampling_strategy = VOO(problem_env, pick_pi, place_pi, explr_p=0.3)
        else:
            sampling_strategy = Uniform(problem_env, pick_pi, place_pi)

    elif domain_name == 'convbelt':
        pick_pi = PickWithoutBaseUnif(problem_env, problem_env.problem['env'].GetRobots()[0],
                                      problem_env.problem['all_region'])
        place_pi = PlaceUnif(problem_env.problem['env'], problem_env.problem['env'].GetRobots()[0],
                             problem_env.obj_region, problem_env.robot_region)

        if sampling_strategy == 'voo':
            sampling_strategy = VOO(problem_env, pick_pi, place_pi, explr_p=0.3)
        else:
            sampling_strategy = Uniform(problem_env, pick_pi, place_pi)

    elif domain_name == 'mover':
        # define and test pick operator
        two_arm_pick_pi = PickWithBaseUnif(problem_env.env, problem_env.env.GetRobots()[0])
        two_arm_place_pi = PlaceUnif(problem_env.env, problem_env.env.GetRobots()[0])

        # todo make MoverVOO
        if sampling_strategy == 'voo':
            sampling_strategy = MoverVOO(problem_env, two_arm_pick_pi, two_arm_place_pi, explr_p=0.3)
        else:
            sampling_strategy = Uniform(problem_env, two_arm_pick_pi, two_arm_place_pi)
    return sampling_strategy


def make_problem_env(domain_name, problem_index):
    if domain_name == 'namo':
        problem_env = NAMO(problem_index)
    elif domain_name == 'convbelt':
        problem_env = ConveyorBelt()
    else:
        problem_env = Mover()
    return problem_env


def get_task_plan(domain_name, problem_env):
    if domain_name == 'namo':
        raise NotImplementedError
    elif domain_name == 'convbelt':
        raise NotImplementedError
    else:
        packing_boxes = problem_env.packing_boxes
        task_plan = [{'region': problem_env.regions['home_region'], 'objects': packing_boxes[0:5]}]
    return task_plan


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-widening_parameter', type=float, default=0.5)
    parser.add_argument('-sampling_strategy', type=str, default='voo')
    parser.add_argument('-problem_idx', type=int, default=0)
    parser.add_argument('-domain', type=str, default='mover')
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-v', action='store_true', default=False)

    print "RANDOM SEED SET", np.random.seed(12)
    print "RANDOM SEED SET", random.seed(12)

    args = parser.parse_args()
    uct_parameter = args.uct
    widening_parameter = args.widening_parameter
    sampling_strategy = args.sampling_strategy
    problem_index = args.problem_idx

    save_dir = make_save_dir(args.domain, uct_parameter, widening_parameter, sampling_strategy)
    problem_env = make_problem_env(args.domain, problem_index)
    sampling_strategy = make_sampling_strategy(args.sampling_strategy, args.domain, problem_env)
    task_plan = get_task_plan(args.domain, problem_env)

    if args.v:
        problem_env.env.SetViewer('qtcoin')

    mcts = MCTS(widening_parameter, uct_parameter, sampling_strategy, problem_env, args.domain, task_plan)
    search_time_to_reward, plan, optimal_score_achieved = mcts.search()
    problem_env.visualize_plan(plan)
    import pdb;pdb.set_trace()

    pickle.dump({'search_time': search_time_to_reward, 'plan': plan, 'pidx': problem_index,
                 'is_optimal_score': optimal_score_achieved},
                open(save_dir + str(problem_index)+'.pkl', 'wb'))
    problem_env.problem['env'].Destroy()
    import pdb;pdb.set_trace()
    openravepy.RaveDestroy()


if __name__ == '__main__':
    main()
