import numpy as np
import copy


class DOOTreeNode:
    def __init__(self, cell_mid_point, cell_min, cell_max, parent_node, distance_fn):
        self.cell_mid_point = cell_mid_point
        self.evaluated_x = None
        self.l_child = None
        self.r_child = None
        self.cell_min = cell_min  # size of the cell
        self.cell_max = cell_max
        self.delta_h = distance_fn(cell_mid_point, self.cell_min) + distance_fn(cell_mid_point, self.cell_min)
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
        self.x_to_node = {}
        self.node_to_update = None

    def create_node(self, cell_mid_point, cell_min, cell_max, parent_node):
        new_node = DOOTreeNode(cell_mid_point, cell_min, cell_max, parent_node, self.distance_fn)
        return new_node

    def add_left_child(self, parent_node):
        left_child_cell_mid_point_x = self.compute_left_child_cell_mid_point(parent_node)
        cell_min, cell_max = self.compute_left_child_cell_limits(parent_node)

        node = self.create_node(left_child_cell_mid_point_x, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'left')

    def add_right_child(self, parent_node):
        right_child_cell_mid_point_x = self.compute_right_child_cell_mid_point(parent_node)
        cell_min, cell_max = self.compute_right_child_cell_limits(parent_node)

        node = self.create_node(right_child_cell_mid_point_x, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'right')

    def find_leaf_with_max_upper_bound_value(self):
        max_upper_bound = -np.inf
        for leaf_node in self.leaves:
            if leaf_node.f_value is None:
                return leaf_node
            if leaf_node.f_value == 'update_me':
                continue
            node_upper_bound = leaf_node.f_value + self.explr_p*leaf_node.delta_h
            if node_upper_bound > max_upper_bound:
                best_leaf = leaf_node
                max_upper_bound = node_upper_bound
        is_node_children_added = not(best_leaf.l_child is None)
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

    def get_next_point_and_node_to_evaluate(self):
        # how can I vary this?
        # I can randomly sample a point?
        is_first_evaluation = self.root is None
        dim_domain = len(self.domain[0])
        if is_first_evaluation:
            #cell_mid_point = (self.domain[1] + self.domain[0]) / 2.0
            cell_mid_point = np.random.uniform(self.domain[0], self.domain[1], (1, dim_domain)).squeeze()
            node = self.create_node(cell_mid_point, self.domain[0], self.domain[1], None)
            self.leaves.append(node)
            self.root = node
        else:
            node = self.find_leaf_with_max_upper_bound_value()
        return node

    def expand_node(self, fval, node):
        if fval == 'update_me':
            self.node_to_update = node
        else:
            self.node_to_update = None

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
        # it all gets stuck here most of the time.
        # This is likely because there are so many self.nodes and that there are so many evaled_x
        # create a mapping between the node to the evaled_x value
        is_in_array = [np.all(np.isclose(target_x_value, a)) for a in evaled_x]
        is_action_included = np.any(is_in_array)
        assert is_action_included, 'action that needs to be updated does not have a value'
        return evaled_y[np.where(is_in_array)[0][0]]

    def update_evaled_values(self, evaled_x, evaled_y, infeasible_reward):
        if len(evaled_x) == 0:
            return

        feasible_idxs = np.array(evaled_y) != infeasible_reward
        evaled_x_to_update = np.array(evaled_x)[feasible_idxs, :]  # only the feasible ones get their f values updated
        evaled_y_to_update = np.array(evaled_y)[feasible_idxs]

        if len(evaled_x_to_update) > 0:
            for l in self.leaves:
                if l.f_value != infeasible_reward and l.f_value != 'update_me':
                    try:
                        l.f_value = self.find_evaled_f_value(l.evaluated_x, evaled_x_to_update, evaled_y_to_update)
                    except:
                        import pdb;pdb.set_trace()

        if self.node_to_update is not None:
            if len(evaled_x_to_update) > 0:
                self.node_to_update.f_value = self.find_evaled_f_value(self.node_to_update.evaluated_x, evaled_x, evaled_y)
            else:
                self.node_to_update.f_value = infeasible_reward

        #for node in self.nodes:
        #    node.f_value = self.find_evaled_f_value(node.evaluated_x, evaled_x, evaled_y)

    @staticmethod
    def add_node_to_tree(node, parent_node, side):
        node.parent = parent_node
        if side == 'left':
            parent_node.l_child = node
        else:
            parent_node.r_child = node

    @staticmethod
    def compute_left_child_cell_mid_point(node):
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (node.cell_min[cutting_dimension] + node.cell_mid_point[cutting_dimension]) / 2.0
        return cell_mid_point

    @staticmethod
    def compute_right_child_cell_mid_point(node):
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (node.cell_max[cutting_dimension] + node.cell_mid_point[cutting_dimension]) / 2.0

        return cell_mid_point

    @staticmethod
    def compute_left_child_cell_limits(node):
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_min = copy.deepcopy(node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_max[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max

    @staticmethod
    def compute_right_child_cell_limits(node):
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_min = copy.deepcopy(node.cell_min)
        cell_min[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max
