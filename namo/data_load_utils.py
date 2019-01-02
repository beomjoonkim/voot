import os
import numpy as np
import cPickle as pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
#from preprocessing_utils import aggregate_data

sys.path.append('../mover_library/')
from utils import clean_pose_data
import time

def process_traj( traj_data ):
  import pdb;pdb.set_trace()

def reshape_data( fc,misc,sumVal):
  fc        = np.array(fc)
  misc      = np.array(misc)
  sumVal    = np.array(sumVal)

  n_data = len(fc)
  if n_data == 0:
    return fc,misc,sumVal

  # reshape data 
  try:
    n_data = len(fc)
    n_konf = max(np.array(fc).shape[1:])
    fc      = fc.reshape((n_data,n_konf,4))
    misc    = misc.reshape((n_data,9))
    sumVal  = sumVal.reshape(n_data,1) # There is a mismatch of n data between sumVal and fc
  except:
    print "In reshape data"
    import pdb;pdb.set_trace()
  return fc,misc,sumVal

def create_bit_encoding_of_konf(n_konf):
  n_helper = int(np.log2(n_konf)+1)
  k_data = np.zeros((n_konf,n_helper))
  idnumb = 1
  for idx in range(n_konf):
    #binstr=bin(idnumb)[2:]
    binstr =  '{0:010b}'.format(idnumb)
    binidx = range(len(binstr))[::-1]
    for k in binidx:
      k_data[idx,k] = int(binstr[k])    
    idnumb+=1
  k_data = k_data.reshape( (n_konf,n_helper))
  return k_data

def convert_collision_vec_to_one_hot(c_data):
  n_konf = c_data.shape[1]
  onehot_cdata = []
  for cvec in c_data:
    one_hot_cvec = np.zeros((n_konf,2))
    for boolean_collision,onehot_collision in zip(cvec,one_hot_cvec):
      onehot_collision[boolean_collision] = 1
    assert( np.all(np.sum(one_hot_cvec,axis=1)==1) )
    onehot_cdata.append(one_hot_cvec)

  onehot_cdata = np.array(onehot_cdata)
  return onehot_cdata

def get_sars_data(data):
  s_cvec         = data[0]
  sprime_cvec    = data[1]
  s_misc         = data[2]
  sprime_misc    = data[3]
  actions        = data[4]
  rewards        = data[5][:,None]
  sumR           = data[6][:,None]
  scores         = data[7][:,None]
  aprimes         = data[8]
  return s_cvec,sprime_cvec,s_misc,sprime_misc,actions,rewards,sumR,scores,aprimes

def get_data_dimensions( pick_data,place_data ):
  pick_s_cvec,_,pick_s_misc,_,pick_actions,_,_,_ = get_sars_data(pick_data)
  place_s_cvec,_,_,_,place_actions,_,_,_         = get_sars_data(place_data)

  dim_misc  = pick_s_misc.shape[-1]
  dim_cvec  = (pick_s_cvec.shape[1],pick_s_cvec.shape[2])
  dim_pick  = pick_actions.shape[-1]
  dim_place = place_actions.shape[-1]
  return dim_misc,dim_cvec,dim_pick,dim_place

def aggregate_place_data( train_dir,fname_keyword ):
  # loops through train files in train dir, loading data from files with fname_keyword in it
  train_dir = train_dir+'/'
  agg_data = {'x_data':[],'s_data':[],'c_data':[],'tobj_data':[],'c0_data':[],'f_data':[]}
  for fdata in os.listdir(train_dir):
    if fdata.find(fname_keyword) == -1: continue
    if fdata.find('aggregated') != -1: continue
    print fdata
    data=pickle.load(open( train_dir + fdata))
    if data[0] == None:continue
    agg_data['x_data'].extend( data[0] )
    agg_data['s_data'].extend( data[1] )
    for d in data[2]:
      agg_data['c0_data'].append( d['c0'] )
      agg_data['c_data'].append( d['c_vec'] )
      agg_data['tobj_data'].append( d['target_o_pose'] )
      agg_data['f_data'].append( d['f_vec'] )
    print len(agg_data['x_data'])
    if len(agg_data['x_data']) > 6000:
      print 'saving',len(agg_data['x_data'])

      agg_data['x_data'] = np.array(agg_data['x_data']).squeeze()
      agg_data['s_data'] = np.array(agg_data['s_data'])[:,None]
      agg_data['c0_data'] = np.array(agg_data['c0_data']).squeeze()
      agg_data['c_data'] = np.array(agg_data['c_data']).squeeze()
      agg_data['tobj_data'] = np.array(agg_data['tobj_data']).squeeze()
      agg_data['f_data']    = np.array(agg_data['f_data']).squeeze()
      
      with open(train_dir+fname_keyword+'_aggregated_data.p','wb') as foutput:
        pickle.dump(agg_data,foutput)
      return agg_data
  

def aggregate_pick_data( train_dir,fname_keyword ):
  # loops through train files in train dir, loading data from files with fname_keyword in it
  train_dir = train_dir+'/'
  agg_data = {'x_data':[],'s_data':[],'c_data':[],'op_data':[],\
              'osh_data':[],'c0_data':[],'tobj_data':[],'f_data':[]}
  for fidx,fdata in enumerate(os.listdir(train_dir)):
    if fdata.find(fname_keyword) == -1: continue
    if fdata.find('aggregated') != -1: continue
    if fdata.find('.p') ==-1: continue
    print fidx,fdata
    data=pickle.load(open( train_dir + fdata))
    if data[0] == None:continue
    agg_data['x_data'].extend( data[0] )
    agg_data['s_data'].extend( data[1] )
    for d in data[2]:
      agg_data['c0_data'].append( d['c0'] )
      agg_data['c_data'].append( d['c_vec'] )
      agg_data['op_data'].append( d['o_pose'] )
      agg_data['osh_data'].append( d['o_shape'] )
      agg_data['tobj_data'].append( d['target_o_pose'] )
      agg_data['f_data'].append( d['f_vec'] )

    n_data = len(agg_data['x_data'])
    print n_data
    if n_data > 5000:
      print 'saving',len(agg_data['x_data'])

      agg_data['x_data'] = np.array(agg_data['x_data']).squeeze()
      agg_data['s_data'] = np.array(agg_data['s_data'])[:,None]
      agg_data['c0_data'] = np.array(agg_data['c0_data']).squeeze()
      agg_data['c_data'] = np.array(agg_data['c_data']).squeeze()
      agg_data['osh_data'] = np.array(agg_data['osh_data']).squeeze()
      agg_data['op_data'] = np.array(agg_data['op_data']).squeeze()
      agg_data['tobj_data'] = np.array(agg_data['tobj_data']).squeeze()
      agg_data['f_data']    = np.array(agg_data['f_data']).squeeze()
      
      with open(train_dir+fname_keyword+'_aggregated_data.p','wb') as foutput:
        pickle.dump(agg_data,foutput)
      return agg_data
"""
def clean_pose_data(pose_data):
  # fixes angle to be between 0 to 2pi
  pose_data[pose_data[:,-1]<0,-1]+=2*np.pi
  assert( np.all(pose_data[:,-1]>=0) and np.all(pose_data[:,-1] <2*np.pi))
"""
def load_place_data( parent_dir,proc_train_data_dir,n_data):
  # setup directories
  scaler_dir = parent_dir + '/scalers/'
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  

  """
  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(n_data_dir):
    os.mkdir(n_data_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)
  """
  # aggregate data if not done so
  if not os.path.exists(proc_train_data_dir+'/place_aggregated_data.p'):
    aggregate_place_data( proc_train_data_dir, 'place' )
  data = pickle.load( open( proc_train_data_dir+'/place_aggregated_data.p','r') )
  # score data
  s_data = data['s_data'][:n_data,:]
  
  # x data
  SCALE_SWITCH = True
  x_data = data['x_data'][:n_data,:]
  clean_pose_data(x_data)
  x_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_x = x_scaler.fit_transform( x_data )

  # curr robot pose data TODO scale it - may be not
  c0_data = data['c0_data'][:n_data,:]
  clean_pose_data(c0_data)
  c0_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  c0_data = c0_scaler.fit_transform(c0_data)

  # target obj pose data TODO scale it
  tobj_data = data['tobj_data'][:n_data,:]
  clean_pose_data(tobj_data)
  tobj_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  tobj_data = tobj_scaler.fit_transform(tobj_data)
 
  # collision vector
  c_data  = data['c_data'][:n_data,:]*1

  # key configurations
  key_configs = np.array(pickle.load( open('./key_configs.p','r')))
  #key_config_idxs = np.where(np.sum(c_data,axis=0)!=0)[0]
  #key_config_idxs = range(0,key_configs.shape[0],2)
  #pickle.dump( key_config_idxs, open('./key_config_idxs_n_'+str(n_data)+'.p','wb'))
  #key_configs = key_configs[key_config_idxs,:]
  n_konf = len(key_configs)
  dim_konf = key_configs.shape[1]

  # filter the c vector
  #c_data = c_data[:,key_config_idxs]
  c_data = convert_collision_vec_to_one_hot( c_data )
  c_scaler = StandardScaler(with_mean=False,with_std=False)

  # fetch vector
  f_data = data['f_data'][:n_data,:]
  #f_data = f_data[:,key_config_idxs]
  f_data = convert_collision_vec_to_one_hot( f_data )
  scaled_c =np.concatenate( [c_data,f_data],axis=-1)
  scaled_c = scaled_c.reshape((n_data,n_konf,np.shape(scaled_c)[-1]))

  # save the scalers
  pickle.dump(x_scaler, open(scaler_dir+'/x_scaler.p','wb'))
  pickle.dump(c0_scaler, open(scaler_dir+'/c0_scaler.p','wb'))
  pickle.dump(tobj_scaler, open(scaler_dir+'/tobj_scaler.p','wb'))
  pickle.dump(c_scaler, open(scaler_dir+'/c_scaler.p','wb'))
  
  # return the scaled data
  # x - robot placement base pose
  # c - collision vector
  # k - key confgs
  # s - score data
  # c0 - robot pick pose (current base pose)
  # o  - target obj pose
  data = {'x':scaled_x,'c':scaled_c,'s':s_data,\
          'c0':c0_data,'o':tobj_data,'key_configs':key_configs}
  scalers={'x_scaler':x_scaler,
           'c_scaler':c_scaler,\
           'c0_scaler':c0_scaler,\
           'o_scaler':tobj_scaler}
  return data,scalers

def load_pick_data( parent_dir,proc_train_data_dir,n_data,n_trial):
  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(n_data_dir):
    os.mkdir(n_data_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)
  if not os.path.exists(proc_train_data_dir+'/pick_aggregated_data.p'):
    aggregate_pick_data( proc_train_data_dir, 'pick' )

  import gc
  stime=time.time()
  datafile = open( proc_train_data_dir+'/pick_aggregated_data.p','rb')
  gc.disable()
  data = pickle.load( datafile )
  print time.time()-stime
  gc.enable()
  datafile.close()
  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  # score data
  s_data = data['s_data'][:n_data,:]

  SCALE_SWITCH = True

  # obj pose data
  opose_data  = data['op_data'][:n_data,:]
  clean_pose_data(opose_data)
  opose_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_opose = opose_scaler.fit_transform( opose_data )

  # x data
  #TODO: Sample x,y of base pose only; 
  x_data = data['x_data'][:n_data,:-1]
  opose_xy = opose_data[:,:-1]
  robot_xy = x_data[:,3:]
  assert( np.all(np.linalg.norm(robot_xy-opose_xy,axis=1)<0.9844) )
  x_data[:,3:] = robot_xy-opose_xy

  """
  x_data = data['x_data'][:n_data,:-1] # first three are grasp params, latter three robot pose
  robot_rel_pose = x_data[:,-2:] - opose_data[:,-2:]
  x_data[:,-2:] = robot_rel_pose
  """
  x_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_x = x_scaler.fit_transform( x_data )

  c_data  = data['c_data'][:n_data,:]*1

  # key configs
  key_configs = np.array(pickle.load( open('./key_configs.p','r')))
  #key_config_idxs = np.where(np.sum(c_data,axis=0)!=0)[0]
  #pickle.dump( key_config_idxs, open('./pick_key_config_idxs_n_'+str(n_data)+'.p','wb'))
  #key_configs = key_configs[key_config_idxs,:]
  n_konf = len(key_configs)
  dim_konf = key_configs.shape[1]

  # collision vector
  #c_data = c_data[:,key_config_idxs]
  c_data = convert_collision_vec_to_one_hot( c_data )
  c_scaler = StandardScaler(with_mean=False,with_std=False)

  # fetch vector
  f_data = data['f_data'][:n_data,:]
  #f_data = f_data[:,key_config_idxs]
  f_data = convert_collision_vec_to_one_hot( f_data )
  scaled_c =np.concatenate( [c_data,f_data],axis=-1)
  scaled_c = scaled_c.reshape((n_data,n_konf,np.shape(scaled_c)[-1]))

  # obj shape data
  oshape_data = data['osh_data'][:n_data,:]
  oshape_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_oshape = oshape_scaler.fit_transform( oshape_data )

  # target obj pose data TODO scale it
  tobj_data = data['tobj_data'][:n_data,:]
  clean_pose_data(tobj_data)
  tobj_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_tobj = tobj_scaler.fit_transform(tobj_data)

  # current robot pose data
  c0_data = data['c0_data'][:n_data,:]
  clean_pose_data(c0_data)
  c0_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_c0 = c0_scaler.fit_transform(c0_data)

  data = {'s':s_data,\
          'x':scaled_x,\
          'key_configs':key_configs,\
          'c':scaled_c,\
          'opose':scaled_opose,\
          'oshape':scaled_oshape,\
          'tobj':scaled_tobj,\
          'c0':scaled_c0}

  scalers={ 'x_scaler':x_scaler,\
            'opose_scaler':opose_scaler,\
            'oshape_scaler':oshape_scaler,\
            'tobj_scaler':tobj_scaler,\
            'c0_scaler':c0_scaler}

  pickle.dump( scalers, open(scaler_dir+'/scalers.p','wb') )
  return data,scalers
 

def load_grasp_data(parent_dir,proc_train_data_dir,n_data,n_trial):
  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(n_data_dir):
    os.mkdir(n_data_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)
  if not os.path.exists(proc_train_data_dir+'/pick_aggregated_data.p'):
    aggregate_pick_data( proc_train_data_dir, 'pick' )
  data = pickle.load( open( proc_train_data_dir+'/pick_aggregated_data.p','r') )

  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  # score data
  s_data = data['s_data'][:n_data,:]
  
  # x data
  SCALE_SWITCH = True
  x_data = data['x_data'][:n_data,:3] # first three are grasp params
  x_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_x = x_scaler.fit_transform( x_data )

  # obj pose data
  opose_data  = data['op_data'][:n_data,:]
  clean_pose_data(opose_data)
  opose_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_opose = opose_scaler.fit_transform( opose_data )

  # relative robot pick base pose
  abs_rpose_data = data['x_data'][:n_data,-3:]
  clean_pose_data(abs_rpose_data)
  rpose_data = abs_rpose_data;
  rpose_data = abs_rpose_data - opose_data 
  rpose_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_rpose = rpose_scaler.fit_transform( rpose_data )
  
  #TODO May be make it really simple
  # given obj shape, predict IR and grasp
  # obj shape data
  oshape_data = data['osh_data'][:n_data,:]
  clean_pose_data(oshape_data)
  oshape_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_oshape = oshape_scaler.fit_transform( oshape_data )

  # which one do I use? Pick!
  key_configs = np.array(pickle.load( open('./pick_key_configs.p','r')))

  data = {
          's':s_data,\
          'x':x_data,\
          'rpose':scaled_rpose,\
          'oshape':scaled_oshape,\
          'opose':scaled_opose,\
          'x_scaler':x_scaler,\
          'rpose_scaler':rpose_scaler,\
          'oshape_scaler':oshape_scaler,\
          'opose_scaler':opose_scaler}
  return data


def load_pbase_data(parent_dir,proc_train_data_dir,n_data,n_trial):
  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(n_data_dir):
    os.mkdir(n_data_dir)
  if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
  if not os.path.exists(scaler_dir):
    os.mkdir(scaler_dir)
  if not os.path.exists(train_results_dir):
    os.mkdir(train_results_dir)
  if not os.path.exists(proc_train_data_dir+'/pick_aggregated_data.p'):
    aggregate_pick_data( proc_train_data_dir, 'pick' )
  data = pickle.load( open( proc_train_data_dir+'/pick_aggregated_data.p','r') )

  n_data_dir = parent_dir+'/n_data_'+str(n_data)
  trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
  scaler_dir = trial_dir + '/scalers/'
  train_results_dir = trial_dir + '/train_results/'

  # score data
  s_data = data['s_data'][:n_data,:]
  
  SCALE_SWITCH = True

  # current robot pose data
  c0_data = data['c0_data'][:n_data,:]
  clean_pose_data(c0_data)
  assert(np.all(c0_data[:,-1] >=0 ) and np.all(c0_data[:,-1]<=2*np.pi))
  c0_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_c0 = c0_scaler.fit_transform(c0_data)

  # collision vector
  c_data  = data['c_data'][:n_data,:]*1

  # key configurations
  #key_config_idxs = np.where(np.sum(c_data,axis=0)!=0)[0]
  #pickle.dump( key_config_idxs, open('./pick_key_config_idxs_n_'+str(n_data)+'.p','wb'))
  key_configs = np.array(pickle.load( open('./key_configs.p','r')))
  assert(np.all(key_configs[:,-1] >=0 ) and np.all(key_configs[:,-1]<=2*np.pi))
  #key_configs = key_configs[key_config_idxs,:]
  n_konf = len(key_configs)
  dim_konf = key_configs.shape[1]

  # filter the c vector
  #c_data = c_data[:,key_config_idxs]
  stime = time.time()
  c_data = convert_collision_vec_to_one_hot( c_data )
  print 'collision vec to one hot time',time.time()-stime
  c_scaler = StandardScaler(with_mean=False,with_std=False)
  scaled_c = c_data
  scaled_c = scaled_c.reshape((n_data,n_konf,np.shape(c_data)[2]))

  # make k 
  k_data = np.tile( key_configs,(n_data,1,1) )  # shape = n_data,n_conf,dim_conf
  k_data = k_data.reshape((n_data,dim_konf*n_konf))
  k_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_k = k_scaler.fit_transform( k_data )
  scaled_k = scaled_k.reshape((n_data,n_konf,3,1))

  # obj shape data - I probably don't need this
  oshape_data = data['osh_data'][:n_data,:]
  clean_pose_data(oshape_data)
  oshape_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_oshape = oshape_scaler.fit_transform( oshape_data )

  # target obj pose data TODO scale it
  tobj_data = data['tobj_data'][:n_data,:]
  clean_pose_data(tobj_data)
  tobj_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  tobj_data = tobj_scaler.fit_transform(tobj_data)
 
  # obj pose data
  opose_data  = data['op_data'][:n_data,:]
  clean_pose_data(opose_data)
  opose_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_opose = opose_scaler.fit_transform( opose_data )

  # x data
  x_data = data['x_data'][:n_data,-3:] # the latter three are abs base pose for picking
  clean_pose_data(x_data)
  x_data[0:2] = x_data[0:2] - opose_data[0:2]         # relative pose
  x_scaler = StandardScaler(with_mean=SCALE_SWITCH,with_std=SCALE_SWITCH)
  scaled_x = x_scaler.fit_transform( x_data )

  # make sure data are transformed
  assert( np.all(np.isclose( x_scaler.inverse_transform(scaled_x) , x_data)))
  assert( np.all(np.isclose( oshape_scaler.inverse_transform(scaled_oshape) , oshape_data)))
  assert( np.all(np.isclose( opose_scaler.inverse_transform(scaled_opose) , opose_data)))
  assert( np.all(np.isclose( c0_scaler.inverse_transform(scaled_c0) , c0_data)))

  pickle.dump(x_scaler, open(scaler_dir+'/x_scaler.p','wb'))
  pickle.dump(c0_scaler, open(scaler_dir+'/c0_scaler.p','wb'))
  pickle.dump(opose_scaler, open(scaler_dir+'/opose_scaler.p','wb'))
  pickle.dump(k_scaler, open(scaler_dir+'/k_scaler.p','wb'))
  pickle.dump(c_scaler, open(scaler_dir+'/c_scaler.p','wb'))
  pickle.dump(tobj_scaler, open(scaler_dir+'/tobj_scaler.p','wb'))
    
  data = {'x':x_data,\
          's':s_data,\
          'k':scaled_k,\
          'c':scaled_c,\
          'c0':scaled_c0,\
          'opose':scaled_oshape,\
          'oshape':scaled_opose,\
          'x_scaler':x_scaler,\
          'c0_scaler':c0_scaler,\
          'c_scaler':c_scaler,\
          'k_scaler':k_scaler,\
          'opose_scaler':opose_scaler,\
          'oshape_scaler':oshape_scaler,\
          'tobj_scaler':tobj_scaler,\
          'tobj':tobj_data,\
          'key_configs':key_configs}
  return data

def format_RL_data( trajs,action_type,n_data=None ):
  all_s_miscs      = []
  all_sprime_miscs = []
  all_s_fc         = []
  all_sprime_fc    = []
  all_actions      = []
  all_rewards      = []
  all_sum_rewards  = []
  all_scores       = []
  all_aprimes      = [] 

  num_transitions = 0
  for i,tau in enumerate(trajs):
    step_idxs = range(len(tau['s_cvec']))
    tau['r'] = [r if r!=10 else 1 for r in tau['r']]  # This is to be turned on

    if len(tau['a'])==0: continue
    s_miscs = []; s_fc = []; rewards = []; actions = []; sum_rewards = []; scores = []
    sprime_miscs = []; sprime_fc = []; aprimes = []
    traj_step_idxs   = []

    for s_cvec,f_vec,s_misc,a,r,step_idx in zip(tau['s_cvec'],\
                                                tau['f_vec'],\
                                                tau['s_misc'],\
                                                tau['a'],\
                                                tau['r'],\
                                                step_idxs):
      if action_type=='pick':
        is_pick_action = a.shape[-1]==6
        if not is_pick_action:
          continue
      else:
        is_place_action = a.shape[-1]==3
        if not is_place_action:
          continue

      # make sprimes
      if step_idx+1 <= step_idxs[-1]:
        sprime_cvec = tau['s_cvec'][step_idx+1]
        sprime_fvec = tau['f_vec'][step_idx+1]
        sprime_misc = tau['s_misc'][step_idx+1]
        sprime_cvec = sprime_cvec.reshape(( s_cvec.shape[0],s_cvec.shape[1],s_cvec.shape[2] ))
        sprime_fvec = sprime_fvec.reshape(( f_vec.shape[0],f_vec.shape[1],f_vec.shape[2] ))
        aprime      = 'pick' if tau['a'][step_idx+1].shape[-1] == 6 else 'place' 
        #if tau['a'][step_idx+1] 
        #aprime     = ["None"]
      else:
        sprime_cvec = np.zeros((1,s_cvec.shape[1],s_cvec.shape[2]))
        sprime_fvec = np.zeros((1,f_vec.shape[1],f_vec.shape[2]))
        sprime_misc = np.zeros((s_misc.shape[0],))
        aprime      = "None"

      # create fc vec
      s_cvec = s_cvec.reshape(( s_cvec.shape[0],s_cvec.shape[1],s_cvec.shape[2] ))
      f_vec  = f_vec.reshape(( f_vec.shape[0],f_vec.shape[1],f_vec.shape[2] ))
      
      # add them to the trajectory 
      sprime_fc.append( np.concatenate([sprime_cvec,sprime_fvec],axis=-1 ) )
      sprime_miscs.append(sprime_misc)
      s_fc.append( np.concatenate([s_cvec,f_vec],axis=-1 ) )
      s_miscs.append(s_misc)
      actions.append(a)
      rewards.append(r)
      sum_rewards.append( np.sum(tau['r'][step_idx:]) )
      scores.append( np.sum(tau['r']) )
      all_aprimes.append(aprime)

    num_transitions += len(actions)
    if len(s_miscs)==0:
      continue

    n_key_confs = s_fc[0].shape[1]
    fc_dim      = s_fc[0].shape[2]
    s_fc        = np.array(s_fc).reshape( len(s_fc),n_key_confs,fc_dim)
    s_miscs     = np.array(s_miscs)

    all_sprime_fc.append(sprime_fc)
    all_sprime_miscs.append(sprime_miscs)
    all_s_miscs.append( np.array(s_miscs) )
    all_s_fc.append( np.array(s_fc) )
    all_actions.append( np.array(actions).squeeze())
    all_sum_rewards.append( np.array([sum_rewards]).squeeze()) 
    all_rewards.append( np.array(rewards).squeeze() )
    all_scores.append( np.array(scores).squeeze() )

  #  combine all of the data
  if len(all_actions) == 0:
    return None
  else:
    all_actions      = np.vstack(all_actions)
    all_sum_rewards  = np.hstack(all_sum_rewards)
    all_rewards      = np.hstack( all_rewards )
    all_scores       = np.hstack( all_scores )
    all_s_fc         = np.vstack(all_s_fc)
    all_s_miscs      = np.vstack(all_s_miscs)
    all_sprime_miscs = np.vstack(all_sprime_miscs)
    all_sprime_fc    = np.vstack(all_sprime_fc).reshape(all_s_fc.shape)

    # limit the number of data points
    if n_data == None:
      idxs   = range(len(all_s_fc))
      n_data = len(all_s_fc)
    return [all_s_fc[:n_data,:],\
            all_sprime_fc[:n_data,:],\
            all_s_miscs[:n_data,:],\
            all_sprime_miscs[:n_data,:],\
            all_actions[:n_data,:],\
            all_rewards[:n_data],\
            all_sum_rewards[:n_data],
            all_scores,all_aprimes[:n_data]]

def stack_traj_data(traj_data,n_data=None):
  if n_data == None:
    idxs = range(len(all_s_fc))
    n_data = len(all_s_fc)
  all_actions      = np.vstack(all_actions)
  all_sum_rewards  = np.hstack(all_sum_rewards)
  all_rewards      = np.hstack( all_rewards )
  all_s_fc         = np.vstack(all_s_fc)
  all_s_miscs      = np.vstack(all_s_miscs)
  all_sprime_miscs = np.vstack(all_sprime_miscs)
  all_sprime_fc    = np.vstack(all_sprime_fc)
  return [ all_s_fc,all_sprime_fc,\
           all_s_miscs,all_sprime_miscs,\
           all_actions,all_sum_rewards,\
           all_rewards]

def get_state_vals( s ):
  cvec   = convert_collision_vec_to_one_hot(np.array([s['c_vec']]))
  fvec   = convert_collision_vec_to_one_hot(np.array([s['f_vec']]))
  fc     = np.concatenate([cvec,fvec],axis=-1)
  c0     = clean_pose_data(np.array(s['c0']).astype(float)).squeeze()
  opose  = clean_pose_data(np.array(s['o_pose'])).squeeze()
  oshape = np.array(s['o_shape']).squeeze()
  misc   = np.r_[c0,opose,oshape]
  return fc,misc[None,:]
  
def load_RL_data( n_data ):
  traj_data = []
  
  num_transitions = 0
  # load key configs
  import time
  n_traj = 0
  stime = time.time()
  for fdata in os.listdir('./processed_train_data/'):
    is_not_train_data =  fdata.find('aggregated') != -1 or\
                         fdata.find('box') != -1 or\
                         fdata.find("RL_train_data_")==-1
    if is_not_train_data:
      continue
 
    try:
      trajs_from_episode = pickle.load(open('./processed_train_data/'+fdata,'r'))
    except:
      print fdata
      continue
    Gs=[g for traj in trajs_from_episode for g in traj['G']] #
    
    solution_exists_in_episode = True in Gs
    if not solution_exists_in_episode:
      continue

    for traj in trajs_from_episode:
      # add konf and misc data to the trajectory data
      contexts  = traj['s'] 
      cvecs     = np.array([s['c_vec']*1 for s in traj['s']] )  
      fvecs     = np.array([s['f_vec']*1 for s in traj['s']] )  
      if len(cvecs)==0: continue

      # create konf data
      cvecs    = convert_collision_vec_to_one_hot( cvecs )   
      fvecs    = convert_collision_vec_to_one_hot( fvecs )   
      traj['s_cvec'] = [ s[None,:] for s in cvecs]
      traj['f_vec']  = [ s[None,:] for s in fvecs]

      # create misc data
      traj['s_misc'] = []
      for idx,con in enumerate(contexts):
        c0     = clean_pose_data(np.array(con['c0']).astype(float)).squeeze()
        opose  = clean_pose_data(np.array(con['o_pose'])).squeeze()
        oshape = np.array(con['o_shape']).squeeze()
        misc = np.r_[c0,opose,oshape]
        traj['s_misc'].append( misc )

      # solution trajectory
      traj_data.append( traj )
      num_transitions+=len(traj['s_cvec'])
    n_traj+=1
    print n_traj,num_transitions,n_data

    if num_transitions>=n_data:
      break
  pick_data  = format_RL_data(traj_data,'pick',n_data)
  place_data = format_RL_data(traj_data,'place',n_data)
  return pick_data,place_data,traj_data

def load_raw_traj_data():
  _,_,traj_data = load_RL_data(10) # for unit testing purpose
  return traj_data

def main():
  pass
if __name__ == '__main__':
  main()


