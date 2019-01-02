from planners.voronoi_mcts import VoronoiMCTS
import sys
from conveyor_belt_env import ConveyorBelt
from generators.PlaceUniform import PlaceUnif
from generators.PickUniform import PickUnif
from planners.mcts_graphics import write_dot_file

sys.path.append('../mover_library/')
from samplers import *
from utils import *

convbelt = ConveyorBelt()
problem = convbelt.problem
pick_pi = PickUnif(convbelt, problem['env'].GetRobots()[0], problem['all_region'])
place_pi = PlaceUnif(problem['env'], problem['env'].GetRobots()[0], problem['loading_region'],
                     problem['all_region'])
explr_parameter = 5/0.9
mcts = VoronoiMCTS(explr_parameter, pick_pi, place_pi, 'voo', convbelt)
mcts.simulate(mcts.s0_node, 0)
import pdb;pdb.set_trace()
write_dot_file(mcts.tree, 0)
convbelt.reset_to_init_state()
mcts.simulate(mcts.s0_node, 0)
write_dot_file(mcts.tree, 1)
import pdb;pdb.set_trace()
