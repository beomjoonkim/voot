import pygraphviz as pgv
import numpy as np


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


def add_line(curr_line, key, value):
    if isinstance(value, list):
        value = value[0]
    if key[0].find('pick') != -1:
        try:
            curr_line += ' (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f): %.2f ' % (
            key[1], key[2], key[3], key[4], key[5], key[6], value)
        except:
            curr_line += 'None'
    else:
        curr_line += ' (%.2f,%.2f,%.2f):%.2f ' % (key[1], key[2], key[3], value)
    return curr_line


def get_node_info_in_string(node):
    Q=''
    N = ''
    parent_action = ''
    reward_history=''

    for key, value in zip(node.Q.keys(), node.Q.values()):
        Q = add_line(Q, key, value)

    for key, value in zip(node.reward_history.keys(), node.reward_history.values()):
        reward_history = add_line(reward_history, key, value)

    for key, value in zip(node.N.keys(), node.N.values()):
        N = add_line(N, key, value)

    if node.parent_action is not None:
        pact = node.parent_action
        operator_name = pact['operator_name']
        if pact is None:
            parent_action = 'None'
        elif operator_name.find('pick') != -1:
            params = np.hstack([pact['base_pose'],pact['grasp_params']])
            parent_action += ' (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) '%( params[3], params[4], params[5],
                                                                   params[0], params[1], params[2])
        else:
            parent_action += ' (%.2f,%.2f,%.2f)' % \
                             (pact['base_pose'][0], pact['base_pose'][1], pact['base_pose'][2])
    else:
        parent_action = 'None'

    info = 'parent_action: '+parent_action + '\n' + \
           'N: ' + N + '\n' + \
           'Nvisited: ' + str(node.Nvisited) + '\n' + \
           'Q: ' + Q + '\n'+ \
           'R history: ' + reward_history
    return info


def recursive_write_tree_on_graph(curr_node, graph):
    # todo:
    #   write the object name and whether the place required the CRP planning
    string_form = get_node_info_in_string(curr_node)
    graph.add_node(string_form)
    if curr_node.is_init_node:
        node = graph.get_node(string_form)
        node.attr['color'] = "red"

    if curr_node.is_goal_node:
        node = graph.get_node(string_form)
        node.attr['color'] = "blue"

    for child_idx, child in enumerate(curr_node.children.values()):
        child_string_form = get_node_info_in_string(child)
        graph.add_edge(string_form, child_string_form)  # connect an edge from parent to child
        edge = graph.get_edge(string_form, child_string_form)
        edge.attr['label'] = child.parent_action['operator_name']
        recursive_write_tree_on_graph(child, graph)
    return


def write_dot_file(tree, file_idx, suffix):
    print ("Writing dot file..")
    graph = pgv.AGraph(strict=False, directed=True)
    graph.node_attr['shape'] = 'box'
    recursive_write_tree_on_graph(tree.root, graph)
    graph.layout(prog='dot')
    graph.draw('./test_results/mcts_evolutions/'+str(file_idx)+'_'+suffix+'.png')  # draw png
    print ("Done!")


if __name__ == '__main__':
    main()