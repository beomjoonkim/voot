from manipulation.bodies.bodies import box_body
from mover_library.utils import set_obj_xytheta, visualize_configs
from problem_environments.minimum_displacement_removal import MinimumDisplacementRemoval
from problem_environments.conveyor_belt_env import ConveyorBelt

import pickle
import argparse


def draw_q_value_rod_for_action(action_idx, action, q_val, penv, maxQ):
    normalized_q = q_val/float(maxQ)
    width = 0.2
    length = 0.2
    height = normalized_q

    new_body = box_body(penv.env, width, length, height,
                        name='q_value_obj%s' % action_idx,
                        color=(1, 0, 0))
    penv.env.Add(new_body)

    base_pose = action.continuous_parameters['base_pose']
    set_obj_xytheta(base_pose, new_body)


def get_penv(args):
    if args.domain == 'convbelt':
        return ConveyorBelt(problem_idx=1)
    else:
        return MinimumDisplacementRemoval(problem_idx=0)


def load_data(args):
    fdir = './test_results/'+args.domain+'_results/visualization_purpose/'
    fname = 'node_idx_' + args.node_idx + '_' + args.algo + '.pkl'
    data = pickle.load(open(fdir+fname, 'r'))
    return data


def visualize_base_poses_and_q_values(q_function, penv):
    infeasible_rwd_compensation = 2
    action_idx = 0

    base_poses = []
    for action, q_val in zip(q_function.keys(), q_function.values()):
        if q_val == -infeasible_rwd_compensation:
            continue
        draw_q_value_rod_for_action(action_idx, action, q_val + infeasible_rwd_compensation, penv, 22)
        action_idx += 1
        base_poses.append(action.continuous_parameters['base_pose'])

    visualize_configs(penv.robot, base_poses, 0.8)

def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='minimum_displacement_removal')
    parser.add_argument('-algo', type=str, default='voo')
    parser.add_argument('-node_idx', type=str, default='0')
    args = parser.parse_args()
    visualization_data = load_data(args)

    penv = get_penv(args)
    penv.env.SetViewer('qtcoin')
    state_saver = visualization_data['saver']
    state_saver.Restore()

    q_function = visualization_data['Q']
    visualize_base_poses_and_q_values(q_function, penv)

    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
