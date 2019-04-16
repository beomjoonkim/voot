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
    if base_pose is not None:
        set_obj_xytheta(base_pose, new_body)


def get_penv(args):
    if args.domain == 'convbelt':
        return ConveyorBelt(problem_idx=0)
    else:
        return MinimumDisplacementRemoval(problem_idx=0)


def load_data(args):
    fdir = './test_results/'+args.domain+'_results/visualization_purpose/with_infeasibles/'
    #fdir = './test_results/'+args.domain+'_results/visualization_purpose/without_infeasibles/'
    fdir = './test_results/'+args.domain+'_results/visualization_purpose/'
    fname = 'node_idx_' + args.node_idx + '_' + args.algo + '.pkl'
    print "Loading ", fdir+fname
    data = pickle.load(open(fdir+fname, 'r'))
    return data


def visualize_base_poses_and_q_values(q_function, penv):
    infeasible_rwd_compensation = 2
    action_idx = 0

    base_poses = []
    for action, q_val in zip(q_function.keys(), q_function.values()):
        #if q_val == -infeasible_rwd_compensation:
        #    continue

        if penv.name == 'convbelt':
            maxQ = 1
        else:
            maxQ = 4
        draw_q_value_rod_for_action(action_idx, action, q_val + infeasible_rwd_compensation, penv, maxQ)
        action_idx += 1
        if action.continuous_parameters['base_pose'] is not None:
            base_poses.append(action.continuous_parameters['base_pose'])
    visualize_configs(penv.robot, base_poses, 0.7)


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
