import time
class TreeNode(object):
  #def __init__(self, sample, features, action, reward, terminal, parent=None):
  def __init__(self, state, sample, rwd, parent=None):
    self.sample = sample 		
    self.state  = state     #saver + placement
    self.parent = parent		# previous state info
    self.rwd = rwd

    #self.reward = reward
    self.children = []			# next state info
    self.solution = False
    self.retried= False
    if parent is not None:
      self.depth = self.parent.depth + 1
      self.parent.children.append(self) # this is where children gets added
    else:
      self.depth = 1
    self.goal_node_flag = False

  def retrace_advantage(self):
    self.solution = True
    if self.parent is None:
      return
    self.parent.advantage = self.parent.reward + (self.start_time - self.parent.start_time)
    self.parent.retrace_advantage()
  def retrace(self):
    if self.parent is None:
      return [self]
    return self.parent.retrace() + [self]
  def __repr__(self):
    #return self.__class__.__name__ + '(' + str(self.sample) + ')'
    #return self.__class__.__name__ + '(' + str(point_from_pose(self.sample)[:2]) + ')'
    return self.__class__.__name__ + '(' + str(self.sample) + ')'

##################################################
