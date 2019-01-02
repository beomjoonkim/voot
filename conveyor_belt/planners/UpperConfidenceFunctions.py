class Domain(object):
  '''
  Domain only supports
  (0) continuous hyper rectangles in R^d
  (1) discrete and finite set of items
  '''
  CONTINUOUS = 0
  DISCRETE = 1
  TYPES = [CONTINUOUS, DISCRETE]

  def __init__(self, space_type, domain):
    assert(space_type in self.TYPES)
    self.space_type = space_type

    # check validity of domain
    if space_type == self.CONTINUOUS:
      domain = np.array(domain)
      assert(domain.ndim == 2)
      assert((domain[1]-domain[0] > 0).all())
    elif space_type == self.DISCRETE:
      assert(type(domain) is list)

    self.domain = domain


class ContinuousObjFcn( Function ):
  def __init__(self,domain):
    self.domain = Domain(space_type=0,domain=domain)
    self.fg=None

  def __call__(self,x):
    raise NotImplemented

class UCB(Function):
  # this is the negative version of GP-UCB (for consistency of minimizing)
  def __init__(self, zeta, gp):
    self.zeta = zeta
    self.gp = gp

  def __call__(self, x):
    mu, var = self.gp.predict(x)
    return -mu - self.zeta*np.sqrt(var)

  def fg(self, x):
    # returns function value and gradient value at x
    mu, var = self.gp.predict(x)
    dmdx, dvdx = self.gp.predictive_gradients(x)
    dmdx = dmdx[0, :, 0]
    dvdx = dvdx[0, :]
    f = -mu - self.zeta*np.sqrt(var)
    g = -dmdx - 0.5*dvdx/np.sqrt(var)
    return f[0, 0], g[0, :]

class ProbImprovement(Function):
  def __init__(self, target_val, gp):
    self.target_val = target_val
    self.gp = gp

  def __call__(self, x):
    mu, var = self.gp.predict(x)

    if np.any(var==0):
      var += 0.00000001
    return (self.target_val-mu)/np.sqrt(var)

  def fg(self, x):
    # returns function value and gradient value at x
    mu, var = self.gp.predict(x)
    dmdx, dvdx = self.gp.predictive_gradients(x)
    dmdx = dmdx[0, :, 0]
    dvdx = dvdx[0, :]
    f = (self.target_val-mu)/np.sqrt(var)
    g = (-np.sqrt(var)*dmdx - 0.5*(self.target_val-mu)*dvdx/np.sqrt(var)) / var
    return f[0, 0], g[0, :]

