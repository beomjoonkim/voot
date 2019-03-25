import pygraphviz as pgv
import numpy as np
import time

pick_failed_node_idx = 0
place_failed_node_idx = 0


def get_most_concrete_root_node(ctree):
    curr_node = ctree.root
    while len(curr_node.covered_k_idxs) == len(curr_node.children[0].covered_k_idxs):
        curr_node = curr_node.children[0]
    return curr_node


def get_constraint_in_string(node):
    string_form = ''
    string_form = str(node.constraint.var_types) + '\n'
    for p in node.constraint.value:
        string_form += str(p) + '\n'
    string_form += str(node.covered_k_idxs)
    return string_form


def get_constraint_diff(parent, child):
    pconst = parent.constraint
    cconst = child.constraint
    diff = []
    c_var_types = [p for p in cconst.var_types.iteritems()]
    p_var_types = [p for p in pconst.var_types.iteritems()]
    for vc in c_var_types:
        if vc not in p_var_types:
            diff.append(vc)

    for pc in cconst.value:
        if pc not in pconst.value:
            diff.append(pc)
    return str(diff)


def add_line(curr_line, action, value):
    is_discrete_node = action.continuous_parameters is None
    global pick_failed_node_idx
    global place_failed_node_idx

    if is_discrete_node:
        if action.type == 'two_arm_pick':
            curr_line += 'pick ' + action.discrete_parameters['object'].GetName() + ': %.2f ' % value
        elif action.type == 'two_arm_place':
            curr_line += 'place ' + action.discrete_parameters['region'].name + ': %.2f ' % value
    else:
        base_pose = action.continuous_parameters['base_pose']
        if action.type == 'two_arm_pick':
            if base_pose is None:
                curr_line += 'pick failed %d: %.2f ' % (pick_failed_node_idx, value)
                pick_failed_node_idx += 1
            else:
                curr_line += 'pick (%.2f,%.2f,%.2f):%.2f ' % (base_pose[0], base_pose[1], base_pose[2], value)
        elif action.type == 'two_arm_place':
            if base_pose is None:
                curr_line += 'place failed %d: %.2f' % (place_failed_node_idx, value)
                place_failed_node_idx += 1
            else:
                curr_line += 'place (%.2f,%.2f,%.2f):%.2f ' % (base_pose[0], base_pose[1], base_pose[2], value)

    return curr_line


def write_parent_action(node, child_idx):
    parent_action = ''
    pact = node.parent_action
    operator_name = pact.type

    parent_action = add_line(parent_action, pact, 1)[:-7]

    """
    is_discrete_node = pact.continuous_parameters is None

    if pact is None:
        parent_action += 'None'
    elif operator_name.find('pick') != -1:
        if pact.continuous_parameters['base_pose'] is not None:
            params = np.hstack([pact['base_pose'], pact['grasp_params']])
            parent_action += ' (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) ' % (params[3], params[4], params[5],
                                                                    params[0], params[1], params[2])
        else:
            parent_action += ' infeasible child' + str(child_idx)

    else:
        if pact['base_pose'] is not None:
            parent_action += ' (%.2f,%.2f,%.2f)' % \
                             (pact['base_pose'][0], pact['base_pose'][1], pact['base_pose'][2])
        else:
            parent_action += ' infeasible child' + str(child_idx)
    """

    return parent_action


def get_node_info_in_string(node, child_idx):
    Q = ''
    N = ''
    reward_history = ''

    for key, value in zip(node.Q.keys(), node.Q.values()):
        Q = add_line(Q, key, value)

    for key, value in zip(node.reward_history.keys(), node.reward_history.values()):
        reward_history = add_line(reward_history, key, np.max(value))

    """
    for key, value in zip(node.N.keys(), node.N.values()):
        N = add_line(N, key, value)
    """

    # write parent action
    if node.parent_action is not None:
        parent_action = write_parent_action(node, child_idx)
    else:
        parent_action = 'None'

    info = 'parent_action: '+parent_action + '\n' + \
           'N: ' + N + '\n' + \
           'Nvisited: ' + str(node.Nvisited) + '\n' + \
           'Q: ' + Q + '\n'+ \
           'R history: ' + reward_history
    return info


def recursive_write_tree_on_graph(curr_node, curr_node_string_form, graph):
    """
    string_form = get_node_info_in_string(curr_node, 0)  # I don't need to call this again if we have a parent
    graph.add_node(string_form)

    if curr_node.is_goal_node:
        node = graph.get_node(string_form)
        node.attr['color'] = "blue"
    """

    graph.add_node(curr_node_string_form)
    node = graph.get_node(curr_node_string_form)
    if curr_node.is_init_node:
        node = graph.get_node(curr_node_string_form)
        node.attr['color'] = "red"

    for child_idx, child in enumerate(curr_node.children.values()):
        child_string_form = get_node_info_in_string(child, child_idx)

        graph.add_edge(curr_node_string_form, child_string_form)
        edge = graph.get_edge(curr_node_string_form, child_string_form)
        edge.attr['label'] = child.parent_action.type
        recursive_write_tree_on_graph(child, child_string_form, graph)
    return


def write_dot_file(tree, file_idx, suffix):
    print ("Writing dot file..")
    graph = pgv.AGraph(strict=False, directed=True)
    graph.node_attr['shape'] = 'box'

    root_node_string_form = get_node_info_in_string(tree.root, 0)
    recursive_write_tree_on_graph(tree.root, root_node_string_form, graph)
    graph.layout(prog='dot')
    graph.draw('./test_results/mcts_search_trees/'+str(file_idx)+'_'+suffix+'.png')  # draw png
    #todo test this graphics file
    print ("Done!")


if __name__ == '__main__':
    main()