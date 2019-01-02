from mcts_tree_node import TreeNode
import numpy as np


class VoronoiTreeNode(TreeNode):
    def __init__(self, exploration_parameter, x_diameter, f_star):
        TreeNode.__init__(self, exploration_parameter)
        self.init_diameter = x_diameter
        self.f_star = f_star
        self.maxR = None

    def is_time_to_sample(self):
        n_children = len(self.children)

        if n_children == 0:
            return True
        else:
            n_children = len(self.children)

        max_q = np.max(self.Q.values())
        min_voronoi_diameter = self.exploration_parameter * self.init_diameter / np.power(2, n_children)  # todo tigher bound?
        print 'Max Q', max_q
        print "Min Voronoi diameter", min_voronoi_diameter
        print "f star", self.f_star
        if max_q + min_voronoi_diameter < self.f_star:
            return True
        return False

    def get_best_action(self):
        n_children = len(self.children)
        if n_children == 0:
            return None

        best_value = -np.inf
        for action, value in zip(self.Q.keys(), self.Q.values()):
            min_voronoi_diameter = self.exploration_parameter * self.init_diameter / np.power(2, n_children)
            voronoi_optimistic_value = value + min_voronoi_diameter
            print 'voroi value:', voronoi_optimistic_value

            if voronoi_optimistic_value > best_value:
                best_action = action
                best_value = voronoi_optimistic_value

        is_pick_action = len(best_action) == 2
        if is_pick_action:
            best_action = tuple((np.array(best_action[0]), np.array(best_action[1])))
        else:
            best_action = np.array([list(best_action)])

        return best_action

    def get_child_max_q(self, action):
        n_children = len(self.children)
        if n_children == 0:
            return 0

        child = self.children[action]
        child_qs = child.Q.values()
        if len(child_qs) == 0:
            return 0
        else:
            return np.max(child_qs)

