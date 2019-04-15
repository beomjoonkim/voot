from manipulation.bodies.bodies import box_body
from mover_library.utils import set_obj_xytheta
from problem_environments.minimum_displacement_removal import MinimumDisplacementRemoval

import pickle


def main():
    # toplot = [child.parent_action.continuous_parameters['base_pose'] for child in
    #              self.s0_node.children.values()]
    # get base poses and their Q-values
    problem_env = MinimumDisplacementRemoval(problem_idx=0)


    """
    visualization_data = pickle.load(
                    open('./minimum_displacement_removal_results/visualization/q_function_for_visualization.pkl', 'r'))

    q_function = visualization_data['q_function']
    actions = q_function.keys()
    state_saver = visualization_data['state_saver']
    import pdb;pdb.set_trace()
    """

    # todo continue from here;
    #   1. generate the visualization q function data
    #   2. try to visualize them

    base_pose = [0, 0, 0] # this represents the base pose
    Qval = 10
    width = 0.1
    length = 0.2
    height = 1
    i = 1
    new_body = box_body(problem_env.env, width, length, height,
                        name='obj%s' % i,
                        color=(0, Qval, 0))
    problem_env.env.Add(new_body)
    set_obj_xytheta(base_pose, new_body)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
