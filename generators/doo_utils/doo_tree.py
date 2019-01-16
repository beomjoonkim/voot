import numpy as np
import copy


class DOOTreeNode:
    def __init__(self, x_value, cell_min, cell_max, parent_node, distance_fn):
        self.x_value = x_value
        self.l_child = None
        self.r_child = None
        self.cell_min = cell_min  # size of the cell
        self.cell_max = cell_max
        self.delta_h = max(distance_fn(x_value, self.cell_min), distance_fn(x_value, self.cell_min))
        self.parent = parent_node
        self.f_value = None

    def update_node_f_value(self, fval):
        self.f_value = fval


class BinaryDOOTree:
    def __init__(self, domain, explr_p, distance_fn):
        self.root = None
        self.leaves = []
        self.nodes = []
        self.domain = domain
        self.nodes = []
        self.distance_fn = distance_fn
        self.explr_p = explr_p

    def create_node(self, x_value, cell_min, cell_max, parent_node):
        new_node = DOOTreeNode(x_value, cell_min, cell_max, parent_node, self.distance_fn)
        return new_node

    def add_left_child(self, parent_node):
        x_value = self.compute_left_child_x_value(parent_node)
        cell_min, cell_max = self.compute_left_child_cell_limits(parent_node)

        node = self.create_node(x_value, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'left')

    def add_right_child(self, parent_node):
        x_value = self.compute_right_child_x_value(parent_node)
        cell_min, cell_max = self.compute_right_child_cell_limits(parent_node)

        node = self.create_node(x_value, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'right')

    def find_leaf_with_max_upper_bound_value(self):
        max_upper_bound = -np.inf
        for leaf_node in self.leaves:
            node_upper_bound = leaf_node.f_value + self.explr_p*leaf_node.delta_h
            #print leaf_node.x_value, leaf_node.f_value, self.explr_p*leaf_node.delta_h,node_upper_bound
            if node_upper_bound > max_upper_bound:
                best_leaf = leaf_node
                max_upper_bound = node_upper_bound
        try:
            is_node_children_added = not(best_leaf.l_child is None)
        except:
            import pdb;pdb.set_trace()
        if is_node_children_added:
            is_left_child_evaluated = best_leaf.l_child.f_value is not None
            is_right_child_evaluated = best_leaf.r_child.f_value is not None
            if not is_left_child_evaluated:
                return best_leaf.l_child
            elif not is_right_child_evaluated:
                return best_leaf.r_child
            else:
                assert False, 'When both children have been evaluated, the node should not be in the self.leaves'
        else:
            return best_leaf

    def get_next_node_to_evaluate(self):
        is_first_evaluation = self.root is None
        if is_first_evaluation:
            x_value = (self.domain[1] + self.domain[0]) / 2.0
            node = self.create_node(x_value, self.domain[0], self.domain[1], None)
            self.leaves.append(node)
            self.root = node
        else:
            node = self.find_leaf_with_max_upper_bound_value()
        return node

    def expand_node(self, fval, node):
        node.update_node_f_value(fval)
        self.nodes.append(node)

        self.add_left_child(node)
        self.add_right_child(node)

        if node.parent is not None:
            is_parent_node_children_all_evaluated = node.parent.l_child.f_value is not None \
                                                    and node.parent.r_child.f_value is not None
            if is_parent_node_children_all_evaluated:
                self.add_to_leaf(node.parent.l_child)
                self.add_to_leaf(node.parent.r_child)

    def add_to_leaf(self, node):
        parent_node = node.parent
        self.leaves.append(node)
        if parent_node in self.leaves:
            self.leaves.remove(parent_node)

    def find_evaled_f_value(self, target_x_value, evaled_x, evaled_y):
        is_in_array = [np.all(np.isclose(target_x_value,a)) for a in evaled_x]
        is_action_included = np.any(is_in_array)
        assert is_action_included, 'action that needs to be updated does not have a value'
        return evaled_y[np.where(is_in_array)[0][0]]

    def update_evaled_values(self, evaled_x, evaled_y):
        for node in self.nodes:
            node.f_value = self.find_evaled_f_value(node.x_value, evaled_x, evaled_y)

    @staticmethod
    def add_node_to_tree(node, parent_node, side):
        node.parent = parent_node
        if side == 'left':
            parent_node.l_child = node
        else:
            parent_node.r_child = node

    @staticmethod
    def compute_left_child_x_value(node):
        x_value = copy.deepcopy(node.x_value)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        x_value[cutting_dimension] = (node.cell_min[cutting_dimension] + node.x_value[cutting_dimension]) / 2.0
        return x_value

    @staticmethod
    def compute_right_child_x_value(node):
        x_value = copy.deepcopy(node.x_value)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        x_value[cutting_dimension] = (node.cell_max[cutting_dimension] + node.x_value[cutting_dimension]) / 2.0
        return x_value

    @staticmethod
    def compute_left_child_cell_limits(node):
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_min = copy.deepcopy(node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_max[cutting_dimension] = node.x_value[cutting_dimension]
        return cell_min, cell_max

    @staticmethod
    def compute_right_child_cell_limits(node):
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_min = copy.deepcopy(node.cell_min)
        cell_min[cutting_dimension] = node.x_value[cutting_dimension]
        return cell_min, cell_max
