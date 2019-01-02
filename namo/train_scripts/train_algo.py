from train_test_utils import *
import tensorflow as tf
import socket

ROOTDIR = './'
def main():
  args = parse_args() 
  weight_dir,scaler_dir,eval_dir,parent_dir = setup_dirs(args)
  pick_data,place_data,traj_data = load_RL_data(args.n_data)

  """
  place_dataset={}
  place_dataset['pose_states'] = place_data[2]
  place_dataset['collision_states'] =place_data[0]
  place_dataset['actions'] = place_data[4]
  pickle.dump(place_dataset,open('object_placement_data.pkl','wb'))
  import pdb;pdb.set_trace()
  """
  session = tf.Session()
  policy  = create_pi(session,\
                      weight_dir,\
                      eval_dir,\
                      args.pi,\
                      args.Qloss,\
                      float(args.d_lr),\
                      float(args.g_lr),\
                      float(args.tau),\
                      int(args.n_score),\
                      float(args.explr_const),\
                      args.architecture)
  pi_name = args.pi
  policy.setup_and_save_scalers(pick_data,\
                                place_data,\
                                scaler_dir=scaler_dir)
  policy.load_scalers(scaler_dir)
  print "Training policy..."
  train_policy(pi_name,pick_data,place_data,traj_data,policy,weight_dir,args.v)

if __name__ == '__main__':
  main()


