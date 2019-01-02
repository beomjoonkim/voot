from conveyor_belt.conveyor_belt_env import ConveyorBelt
from planners.mcts import MCTS
from planners.voronoi_mcts import VoronoiMCTS
from generators.PlaceUniform import PlaceUnif
from generators.PickUniform import ConvBeltPickUnif

import argparse
import cPickle as pickle
import os
import openravepy

def make_save_dir(planner, uct_parameter, widening_parameter, sampling_strategy):
    save_dir = './test_results/convbelt/'+planner+'_results/uct_'+ str(uct_parameter) + '_widening_' \
               + str(widening_parameter) + '_'+sampling_strategy+'/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return save_dir


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-widening_parameter', type=float, default=0.49)
    parser.add_argument('-sampling_strategy', type=str, default='voo')
    parser.add_argument('-problem_index', type=int, default=0)
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-v', action='store_true', default=False)

    args = parser.parse_args()
    uct_parameter = args.uct
    widening_parameter = args.widening_parameter
    sampling_strategy = args.sampling_strategy
    problem_index = args.problem_index

    save_dir = make_save_dir(args.planner, uct_parameter, widening_parameter, sampling_strategy)

    convbelt = ConveyorBelt()
    problem = convbelt.problem
    if args.v:
        convbelt.env.SetViewer('qtcoin')

    pick_pi = ConvBeltPickUnif(convbelt, problem['env'].GetRobots()[0], problem['all_region'])
    place_pi = PlaceUnif(problem['env'], problem['env'].GetRobots()[0], problem['loading_region'],
                         problem['all_region'])

    if args.planner == 'mcts':
        mcts = MCTS(widening_parameter, uct_parameter, pick_pi, place_pi,  sampling_strategy, convbelt)
    else:
        mcts = VoronoiMCTS(pick_pi, place_pi, 'voo', convbelt)

    search_time_to_reward, plan = mcts.search()

    pickle.dump({'search_time': search_time_to_reward, 'plan': plan},
                open(save_dir + str(problem_index)+'.pkl', 'wb'))
    convbelt.problem['env'].Destroy()
    openravepy.RaveDestroy()


if __name__ == '__main__':
    main()
