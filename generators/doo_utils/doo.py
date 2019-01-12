import numpy as np


def distance(x, y):
    return abs(x - y)


class DOOTreeNode:
    def __init__(self, x_value, f_value, min_x, max_x):
        self.x_value = x_value
        self.f_value = f_value
        self.l_children = []
        self.r_children = []
        self.min_x = min_x  # size of the cell
        self.max_x = max_x
        self.delta_h = distance(x_value, min_x)
        assert np.isclose(self.delta_h, distance(x_value, max_x))


class BinaryDOOTree:
    # one dimensional DOO
    def __init__(self, domain):
        self.root = None
        self.leaves = []
        self.domain = domain

    def add_left(self, parent_node, fval):
        # todo get the mid point of the largest dimension
        x_value = (parent_node.x_value + parent_node.min_x) / 2.0  # get a mid-point

        node_min_x = parent_node.min_x
        node_max_x = parent_node.x_value

        node = DOOTreeNode(x_value, fval, node_min_x, node_max_x)

        parent_node.l_children.append(node)
        self.leaves.append(node)
        if parent_node in self.leaves:
            self.leaves.remove(parent_node)

        return x_value

    def add_right(self, parent_node, fval):
        x_value = (parent_node.x_value + parent_node.max_x) / 2.0  # get a mid-point

        node_min_x = parent_node.x_value
        node_max_x = parent_node.max_x

        node = DOOTreeNode(x_value, fval, node_min_x, node_max_x)

        parent_node.r_children.append(node)
        self.leaves.append(node)
        if parent_node in self.leaves:
            self.leaves.remove(parent_node)

        return x_value

    def choose_next_point(self):
        is_first_evaluation = self.root == None
        if is_first_evaluation:
            # todo get the midpoint as a
            self.domain[1] - self.domain[0]
            pass

        max_upper_bound = -np.inf
        for leaf_node in self.leaves:
            node_upper_bound = leaf_node.f_value + leaf_node.delta_h
            # print leaf_node.f_value, leaf_node.x_value, leaf_node.delta_h
            if node_upper_bound > max_upper_bound:
                best_leaf = leaf_node
                max_upper_bound = node_upper_bound
        # print 'End of leaves'
        return best_leaf
