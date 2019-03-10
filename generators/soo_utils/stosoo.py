import numpy as np
import copy
from soo_tree import SOOTreeNode,BinarySOOTree


class StoSOOTreeNode(SOOTreeNode):
    def __init__(self, cell_mid_point, cell_min, cell_max, height, parent_node):
        SOOTreeNode.__init__(self, cell_mid_point, cell_min, cell_max, height, parent_node)
        self.sum_f_values = 0
        self.n_evaluations = 0
        self.f_value = 0
        self.height = height

    def update_node_f_value(self, fval):
        self.sum_f_values += fval
        self.n_evaluations += 1
        self.f_value = self.sum_f_values / self.n_evaluations

    def compute_width(self):
        raise NotImplementedError


# note bmax is used as bmax
class BinaryStoSOOTree(BinarySOOTree):
    def __init__(self, confidence_parameter, widening_parameter, total_evaluations, domain):
        # widening parameter - k  in the papper
        # confidence parameter - delta in the paper
        # total_evaluations - n in the paper
        BinarySOOTree.__init__(self, domain)
        self.widening_parameter = widening_parameter
        self.confidence_parameter = confidence_parameter
        self.total_evalutations = total_evaluations
        self.bmax = -np.inf

    def get_best_node(self):
        best_f_value = -np.inf
        best_node = self.nodes[0]
        for node in self.nodes:
            if node in self.leaves:
                continue
            if best_f_value < node.f_value:
                best_f_value = node.f_value
                best_node = node
        return best_node

    @staticmethod
    def create_node(cell_mid_point, cell_min, cell_max, parent_node):
        if parent_node is None:
            height = 0
        else:
            height = parent_node.height + 1
        new_node = StoSOOTreeNode(cell_mid_point, cell_min, cell_max, height, parent_node)
        return new_node

    def compute_width(self, best_leaf):
        n_visited = best_leaf.n_evaluations
        width_numerator = np.log(self.total_evalutations * self.widening_parameter / self.confidence_parameter)
        width_denominator = 2*n_visited
        if width_denominator == 0:
            width = np.inf
        else:
            width = np.sqrt(width_numerator / width_denominator)
        return width

    def find_leaf_with_max_value_at_given_height(self, height):
        leaves = self.get_leaves_at_height(height)
        if len(leaves) == 0:
            return None

        leaf_b_values = [l.f_value + self.compute_width(l) for l in leaves]
        best_leaf = leaves[np.argmax(leaf_b_values)]

        if best_leaf.f_value >= self.bmax:
            self.bmax = best_leaf.f_value + self.compute_width(best_leaf)
            is_node_children_added = not (best_leaf.l_child is None)
            if is_node_children_added:
                is_left_child_evaluated_k_times = best_leaf.l_child.n_evaluations >= self.widening_parameter
                is_right_child_evaluated_k_times = best_leaf.r_child.n_evaluations >= self.widening_parameter
                if not is_left_child_evaluated_k_times:
                    return best_leaf.l_child
                elif not is_right_child_evaluated_k_times:
                    return best_leaf.r_child
                else:
                    assert False, 'When both children have been evaluated, the node should not be in the self.leaves'
            else:
                return best_leaf
        else:
            return None

    def get_leaves_at_height(self, height):
        return [l for l in self.leaves if l.height == height]

    def get_next_point_and_node_to_evaluate(self):
        is_first_evaluation = self.root is None
        dim_domain = len(self.domain[0])

        if is_first_evaluation:
            cell_mid_point = np.random.uniform(self.domain[0], self.domain[1], (1, dim_domain)).squeeze()
            node = self.create_node(cell_mid_point, self.domain[0], self.domain[1], None)
            self.leaves.append(node)
            self.root = node
        else:
            node = self.find_leaf_node_whose_value_is_greater_than_bmax()

        return node

    def find_leaf_node_whose_value_is_greater_than_bmax(self):
        node = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
        no_node_exceeds_bmax = node is None
        while no_node_exceeds_bmax and self.tree_traversal_height <= self.tree_height:
            self.tree_traversal_height += 1
            node = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
            no_node_exceeds_bmax = node is None

        if no_node_exceeds_bmax:
            # it might come here without finding the leaf node. Reset self.bmax in this case
            self.bmax = -np.inf
            self.tree_traversal_height = 0
            node = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
            no_node_exceeds_bmax = node is None
            while no_node_exceeds_bmax and self.tree_traversal_height <= self.tree_height:
                self.tree_traversal_height += 1
                node = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
                no_node_exceeds_bmax = node is None

        return node

    def expand_node(self, fval, node):
        # if it has not been evaluated self.widening_parameter number of times, then do not add children
        node.update_node_f_value(fval)

        if node not in self.nodes:
            self.nodes.append(node)

        print 'n_evaluations %d, and widening parameter %d' % (node.n_evaluations, self.widening_parameter)
        if node.n_evaluations >= self.widening_parameter:
            print 'n_evals reached limit, adding children and initializing their values'
            self.add_left_child(parent_node=node)
            self.add_right_child(parent_node=node)
            not_root_node = node.parent is not None
            if not_root_node:
                self.add_parent_children_to_leaves(node)

    def add_parent_children_to_leaves(self, node):
        is_parent_node_children_all_evaluated = node.parent.l_child.n_evaluations >= self.widening_parameter and \
                                                node.parent.r_child.n_evaluations >= self.widening_parameter

        if is_parent_node_children_all_evaluated:
            # note that parent is not a leaf until its children have been evaluated
            self.add_to_leaf(node.parent.l_child)
            self.add_to_leaf(node.parent.r_child)
            self.tree_traversal_height += 1
            self.tree_height += 1

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
