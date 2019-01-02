import numpy as np

def distance(x,y):
    return abs(x-y)


class DOO_tree_node:
    def __init__(self, x_value, f_value, min_x, max_x):
        self.x_value = x_value
        self.f_value = f_value
        self.l_children = []
        self.r_children = []
        self.min_x = min_x # cell size
        self.max_x = max_x
        self.delta_h = distance(x_value, min_x)
        assert np.isclose(self.delta_h,distance(x_value,max_x))


class binary_DOO_tree:
    def __init__(self, root_node):
        self.root = root_node
        self.leaves = [root_node]

    def add_left(self, f, parent_node):
        x_value = (parent_node.x_value + parent_node.min_x) / 2.0  # get a mid-point

        node_min_x = parent_node.min_x
        node_max_x = parent_node.x_value

        if x_value < node_min_x or x_value > node_max_x:
            import pdb;pdb.set_trace()

        node = DOO_tree_node(x_value, f(x_value), node_min_x, node_max_x)

        parent_node.l_children.append(node)
        self.leaves.append(node)
        if parent_node in self.leaves:
            self.leaves.remove(parent_node)

        return x_value, f(x_value)

    def add_right(self, f, parent_node):
        x_value = (parent_node.x_value + parent_node.max_x) / 2.0 # get a mid-point

        node_min_x = parent_node.x_value
        node_max_x = parent_node.max_x

        if x_value < node_min_x or x_value > node_max_x:
            import pdb;pdb.set_trace()

        node = DOO_tree_node(x_value, f(x_value), node_min_x, node_max_x)

        parent_node.r_children.append(node)
        self.leaves.append(node)
        if parent_node in self.leaves:
            self.leaves.remove(parent_node)

        return x_value, f(x_value)

    def choose_leaf_node_to_expand(self):
        max_upper_bound = -np.inf
        for leaf_node in self.leaves:
            node_upper_bound = leaf_node.f_value + leaf_node.delta_h
            #print leaf_node.f_value, leaf_node.x_value, leaf_node.delta_h
            if node_upper_bound > max_upper_bound:
                best_leaf = leaf_node
                max_upper_bound = node_upper_bound
        #print 'End of leaves'
        return best_leaf
