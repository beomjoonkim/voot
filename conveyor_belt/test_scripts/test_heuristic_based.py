from conveyor_belt_env import ConveyorBelt
from planners.mcts import MCTS
from planners.voronoi_mcts import VoronoiMCTS
from planners.astar_variant import AStarVariant
from generators.PlaceUniform import PlaceUnif
from generators.PickUniform import PickUnif

from openravepy import *

import argparse
import cPickle as pickle
import os


def write_opening():
    with open('mcts_planner_results.txt', 'a') as fopen:
        fopen.write('Widening parameter %.2f, UCT parameter %.2f\n' % (widening_parameter, uct_parameter))


def write_results(search_time, problem_index):
    with open('mcts_planner_results.txt', 'a') as fopen:
        fopen.write('%d search_time, %.5f\n' % (problem_index, search_time))


def make_save_dir(planner, uct_parameter, widening_parameter, sampling_strategy):
    save_dir = './'+planner+'_results/uct_'+ str(uct_parameter) + '_widening_' + str(widening_parameter) + \
               '_'+sampling_strategy+'/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return save_dir


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-sampling_strategy', type=str, default='voo')
    parser.add_argument('-problem_index', type=int, default=0)
    parser.add_argument('-v', action='store_true', default=False)

    args = parser.parse_args()
    sampling_strategy = args.sampling_strategy

    convbelt = ConveyorBelt()
    problem = convbelt.problem
    if args.v:
        convbelt.env.SetViewer('qtcoin')

    pick_pi = PickUnif(convbelt, problem['env'].GetRobots()[0], problem['all_region'])
    place_pi = PlaceUnif(problem['env'], problem['env'].GetRobots()[0], problem['loading_region'],
                         problem['all_region'])

    astar_variant = AStarVariant(pick_pi, place_pi, sampling_strategy, convbelt)

    search_time_to_reward, plan = astar_variant.search()

    convbelt.problem['env'].Destroy()
    RaveDestroy()


if __name__ == '__main__':
    main()
