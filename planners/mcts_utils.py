import numpy as np


def make_action_hashable(action):
    operator_name = action['operator_name']
    hashable_action = [operator_name]
    if operator_name == 'two_arm_pick':
        if action['g_config'] is None:
            hashable_action += 'None'
        else:
            hashable_action += action['grasp_params'].tolist()
            hashable_action += action['base_pose'].tolist()
            hashable_action += action['g_config'][0].tolist()
            hashable_action += action['g_config'][1].tolist()

    elif operator_name == 'two_arm_place':
        hashable_action += action['base_pose'].tolist()

    elif operator_name == 'one_arm_pick':
        #raise NotImplementedError
        if action['grasp_params'] is None:
            hashable_action += 'None'
        else:
            hashable_action += action['grasp_params'].tolist()
            hashable_action += action['base_pose'].tolist()
            hashable_action += action['g_config'].tolist()
    elif operator_name == 'one_arm_place':
        hashable_action += action['base_pose'].tolist()
        hashable_action += action['g_config'].tolist()

    return tuple(hashable_action)


def make_action_executable(action):
    operator_name = action[0]
    executable_action = {'operator_name': operator_name}
    if operator_name == 'two_arm_pick':
        assert len(action) == 22, 'Only handles rightarm torso and left hand pick'
        executable_action['grasp_params'] = np.array(action[1:4])
        executable_action['base_pose'] = np.array(action[4:7])
        executable_action['g_config'] = [np.array(action[7:14])]
        executable_action['g_config'].append(np.array(action[14:]))

    elif operator_name == 'two_arm_place':
        executable_action['base_pose'] = np.array(action[1:4])

    elif operator_name == 'one_arm_pick':
        executable_action['grasp_params'] = np.array(action[1:4])
        executable_action['base_pose'] = np.array(action[4:7])
        executable_action['g_config'] = np.array(action[7:])

    elif operator_name == 'one_arm_place':
        executable_action['base_pose'] = np.array(action[1:4])
        executable_action['g_config'] = np.array(action[4:])

    return executable_action


def is_action_hashable(action):
    return isinstance(action, tuple)
