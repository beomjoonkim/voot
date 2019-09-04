import numpy as np
import copy

from generators.gpucb_utils.gp import StandardContinuousGP
from generators.gpucb_utils.functions import UCB, Domain
from generators.gpucb_utils.bo import BO


class BamSOOTreeNode:
    def __init__(self, cell_mid_point, cell_min, cell_max, height, parent_node, ucb, lcb):
        self.cell_mid_point = cell_mid_point
        self.evaluated_x = None
        self.l_child = None
        self.r_child = None
        self.cell_min = cell_min  # size of the cell
        self.cell_max = cell_max
        self.delta_h = 0
        self.parent = parent_node
        self.f_value = None
        self.height = height
        self.ucb = ucb
        self.lcb = lcb

    def update_node_f_value(self, fval):
        self.f_value = fval


class BamBinarySOOTree:
    def __init__(self, domain, explr_p):
        self.root = None
        self.leaves = []
        self.nodes = []
        self.domain = domain
        self.nodes = []
        self.x_to_node = {}
        self.vmax = -np.inf
        self.tree_traversal_height = 0
        self.tree_height = 0
        dim_x = len(self.domain[0])
        self.gp = StandardContinuousGP(dim_x)
        self.fplus = None
        self.explr_p = explr_p
        self.N = 1

    def create_node(self, cell_mid_point, cell_min, cell_max, parent_node):
        if parent_node is None:
            height = 0
        else:
            height = parent_node.height + 1
        mu, var = self.gp.predict(cell_mid_point)

        running_shekel_3d = True
        if running_shekel_3d:
            Bn = self.explr_p
        else:
            Bn = np.sqrt(2 * np.log((np.pi*np.pi*self.N*self.N) / (6.0 * self.explr_p) ))

        print Bn*var, self.N
        ucb = mu + Bn*var
        lcb = mu - Bn*var
        new_node = BamSOOTreeNode(cell_mid_point, cell_min, cell_max, height, parent_node, ucb, lcb)
        return new_node

    def add_left_child(self, parent_node):
        left_child_cell_mid_point_x = self.compute_left_child_cell_mid_point(parent_node)
        cell_min, cell_max = self.compute_left_child_cell_limits(parent_node)

        # evaluate U_N(x) of the node here

        node = self.create_node(left_child_cell_mid_point_x, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'left')

    def add_right_child(self, parent_node):
        right_child_cell_mid_point_x = self.compute_right_child_cell_mid_point(parent_node)
        cell_min, cell_max = self.compute_right_child_cell_limits(parent_node)

        # evaluate U_N(x) of the node here

        node = self.create_node(right_child_cell_mid_point_x, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'right')

    def find_leaf_with_max_value_at_given_height(self, height):
        leaves = self.get_leaves_at_height(height)
        if len(leaves) == 0:
            return None, 'DontEval'

        leaf_values = [l.f_value for l in leaves]
        best_leaf = leaves[np.argmax(leaf_values)]

        # Here, is the for-loop implementation in BamSOO.
        #   If U_N(x) value is greater than f_plus, then evaluate this point and evaluate it. Update GP. Set g(x) = f(x)
        #   Otherwise, set g(x) = L_N(x)
        #   Okay, in my code, I would just to return the point if U_N(x) > f+. There is no g(x), but only f_value.
        #   So, setting node.f_value = L_N(x) is same is g(x) = L_N(x)
        if best_leaf.f_value >= self.vmax:
            self.vmax = best_leaf.f_value  # Line 24 of BamSOO
            is_node_children_added = not(best_leaf.l_child is None)
            self.N += 1
            if is_node_children_added:
                # I need to check U_N value of the chosen child
                is_left_child_evaluated = best_leaf.l_child.f_value is not None
                is_right_child_evaluated = best_leaf.r_child.f_value is not None

                if not is_left_child_evaluated:
                    node_to_eval = best_leaf.l_child
                elif not is_right_child_evaluated:
                    node_to_eval = best_leaf.r_child
                else:
                    assert False, 'When both children have been evaluated, the node should not be in the self.leaves'
            else:
                # I need to check U_N value of the best_leaf
                node_to_eval = best_leaf

            # lines 12-17 of BamSOO
            #print node_to_eval, self.N, node_to_eval.ucb, self.fplus
            if node_to_eval.ucb >= self.fplus:
                return node_to_eval, 'Eval'
            else:
                node_to_eval.f_value = node_to_eval.lcb
                if node_to_eval.f_value > self.fplus:  # line 19-20 of BamSOO
                    self.fplus = node_to_eval.f_value
                return node_to_eval, 'DontEval'
        else:
            return None, 'DontEval'

    def get_leaves_at_height(self, height):
        return [l for l in self.leaves if l.height == height]

    def get_next_point_and_node_to_evaluate(self):
        is_first_evaluation = self.root is None
        dim_domain = len(self.domain[0])

        # Here, check the children's U_N values
        if is_first_evaluation:
            cell_mid_point = np.random.uniform(self.domain[0], self.domain[1], (1, dim_domain)).squeeze()
            node = self.create_node(cell_mid_point, self.domain[0], self.domain[1], None)
            self.leaves.append(node)
            self.root = node
            node_eval_flag = 'Eval'
        else:
            node, node_eval_flag = self.find_leaf_node_whose_value_is_greater_than_vmax()

        # todo
        #   I need to take care of the case in which there is no node
        #   that has ucb value higher than fplus. In such case, we associate
        #   its lcb value as its f_value, and then expand its children.
        #   The gp is not updated.
        while node_eval_flag == 'DontEval':
            self.nodes.append(node)
            self.add_left_child(parent_node=node)
            self.add_right_child(parent_node=node)

            not_root_node = node.parent is not None
            if not_root_node:
                self.add_parent_children_to_leaves(node)

            node, node_eval_flag = self.find_leaf_node_whose_value_is_greater_than_vmax()
            self.fplus = -np.inf
        return node

    def find_leaf_node_whose_value_is_greater_than_vmax(self):
        node, flag = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
        no_node_exceeds_vmax = node is None
        while no_node_exceeds_vmax and self.tree_traversal_height <= self.tree_height:
            self.tree_traversal_height += 1
            node, flag = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
            no_node_exceeds_vmax = node is None

        if no_node_exceeds_vmax:
            # it might come here without finding the leaf node. Reset self.vmax in this case
            self.vmax = -np.inf
            self.tree_traversal_height = 0
            node, flag = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
            no_node_exceeds_vmax = node is None
            while no_node_exceeds_vmax and self.tree_traversal_height <= self.tree_height:
                self.tree_traversal_height += 1
                node, flag = self.find_leaf_with_max_value_at_given_height(self.tree_traversal_height)
                no_node_exceeds_vmax = node is None

        return node, flag

    def expand_node(self, fval, node, evaled_x, evaled_y):
        node.update_node_f_value(fval)

        # lines 15 in Algorithm 3 of BamSOO
        # Update my gp here
        self.gp.update(evaled_x, evaled_y, is_bamsoo=True)

        # lines 19-20 in Algorithm 3 of BamSOO
        if self.fplus is None or fval > self.fplus:
            self.fplus = fval

        self.nodes.append(node)

        self.add_left_child(parent_node=node)
        self.add_right_child(parent_node=node)

        not_root_node = node.parent is not None
        if not_root_node:
            self.add_parent_children_to_leaves(node)

    def add_parent_children_to_leaves(self, node):
        is_parent_node_children_all_evaluated = node.parent.l_child.f_value is not None \
                                                and node.parent.r_child.f_value is not None
        if is_parent_node_children_all_evaluated:
            # note that parent is not a leaf until its children have been evaluated
            self.add_to_leaf(node.parent.l_child)
            self.add_to_leaf(node.parent.r_child)
            self.tree_traversal_height += 1  # increment the current height only when we evaluated the current node fully
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
