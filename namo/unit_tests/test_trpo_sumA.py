from data_load_utils import *
from train_test_utils import *
import tensorflow as tf
import unittest

class TestCombinedPiSumA( unittest.TestCase ):
  def test(self):
    args = parse_args()
    args.pi = 'gail'
    weight_dir,scaler_dir,eval_dir,parent_dir = setup_dirs(args)
    session = tf.Session()
    pi  = create_pi(session,\
                      weight_dir,\
                      eval_dir,\
                      args.pi,\
                      args.Qloss,\
                      float(args.d_lr),\
                      float(args.g_lr),\
                      float(args.tau),\
                      int(args.n_score),\
                      float(args.explr_const))
    pi.load_scalers(scaler_dir)
    traj_data = pickle.load( open('./traj.pkl','r') )
    pick_sumA,place_sumA = pi.compute_sum_advantage(traj_data)

    _,_,traj_data = load_RL_data(10)
    pick_sumA,place_sumA = pi.compute_sum_advantage(traj_data)


    
if __name__ == '__main__':
  unittest.main()
  
