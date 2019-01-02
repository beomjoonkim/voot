# misc vector slices
# misc_vector = [c0,opose,oshape]
# misc = np.r_[c0,opose,oshape]
def slice_c0(x):
  return x[:,0:3]

def slice_oxy(x):
  return x[:,3:5]

def slice_shape(x):
  return x[:,6:]

def slice_c0_opose(x):
  return x[:,0:6]

# pick action vector slices
# action = np.hstack([theta,height_portion,depth_portion,x,y,th])[None,:]
def slice_grasp(x):
  return x[:,0:3]

def slice_rxy(x):
  return x[:,3:5]

def slice_rth(x):
  return x[:,5:]
