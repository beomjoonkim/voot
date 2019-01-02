from data_load_utils import *
from train_test_utils import *
import tensorflow as tf
import unittest

class TestCorrectPiArguments( unittest.TestCase ):
  def test_soap_params(self):
    args = parse_args()
    args.pi = 'soap'
    pick_data,place_data,traj_data = load_RL_data(args.n_data)
    weight_dir,scaler_dir,eval_dir,parent_dir,otherpi_dir = setup_dirs(args)
    session = tf.Session()
    policy  = create_pi(session,\
                      pick_data,\
                      place_data,\
                      weight_dir,\
                      eval_dir,\
                      args.pi,\
                      args.Qloss,\
                      float(args.d_lr),\
                      float(args.g_lr),\
                      float(args.tau),\
                      int(args.n_score),\
                      float(args.explr_const))
    
    self.assertEqual( args.tau , policy.pick_pi.tau)
    self.assertEqual( args.tau , policy.place_pi.tau)
    self.assertEqual( args.explr_const , policy.pick_pi.explr_const)
    self.assertEqual( args.explr_const , policy.place_pi.explr_const)
    self.assertEqual( args.d_lr , policy.pick_pi.d_lr)
    self.assertEqual( args.d_lr , policy.place_pi.d_lr)
    self.assertEqual( args.g_lr , policy.pick_pi.g_lr)
    self.assertEqual( args.g_lr , policy.place_pi.g_lr)

  def test_gail_params(self):
    args = parse_args()
    args.pi = 'gail'
    pick_data,place_data,traj_data = load_RL_data(args.n_data)
    weight_dir,scaler_dir,eval_dir,parent_dir,otherpi_dir = setup_dirs(args)
    session = tf.Session()
    policy  = create_pi(session,\
                      pick_data,\
                      place_data,\
                      weight_dir,\
                      eval_dir,\
                      args.pi,\
                      args.Qloss,\
                      float(args.d_lr),\
                      float(args.g_lr),\
                      float(args.tau),\
                      int(args.n_score),\
                      float(args.explr_const))
    
    self.assertEqual( args.tau , policy.pick_pi.tau)
    self.assertEqual( args.tau , policy.place_pi.tau)
    self.assertEqual( args.explr_const , policy.pick_pi.explr_const)
    self.assertEqual( args.explr_const , policy.place_pi.explr_const)
    self.assertEqual( args.d_lr , policy.pick_pi.d_lr)
    self.assertEqual( args.d_lr , policy.place_pi.d_lr)
    self.assertEqual( args.g_lr , policy.pick_pi.g_lr)
    self.assertEqual( args.g_lr , policy.place_pi.g_lr)

  def test_trpo_params(self):
    args = parse_args()
    args.pi = 'trpo'
    pick_data,place_data,traj_data = load_RL_data(args.n_data)
    weight_dir,scaler_dir,eval_dir,parent_dir,otherpi_dir = setup_dirs(args)
    session = tf.Session()
    policy  = create_pi(session,\
                      pick_data,\
                      place_data,\
                      weight_dir,\
                      eval_dir,\
                      args.pi,\
                      args.Qloss,\
                      float(args.d_lr),\
                      float(args.g_lr),\
                      float(args.tau),\
                      int(args.n_score),\
                      float(args.explr_const))
    
    self.assertEqual( args.tau , policy.pick_pi.tau)
    self.assertEqual( args.tau , policy.place_pi.tau)
    self.assertEqual( args.explr_const , policy.pick_pi.explr_const)
    self.assertEqual( args.explr_const , policy.place_pi.explr_const)
    self.assertEqual( args.d_lr , policy.pick_pi.d_lr)
    self.assertEqual( args.d_lr , policy.place_pi.d_lr)
    self.assertEqual( args.g_lr , policy.pick_pi.g_lr)
    self.assertEqual( args.g_lr , policy.place_pi.g_lr)
  def test_ddpg_params(self):
    args = parse_args()
    args.pi = 'ddpg'
    pick_data,place_data,traj_data = load_RL_data(args.n_data)
    weight_dir,scaler_dir,eval_dir,parent_dir,otherpi_dir = setup_dirs(args)
    session = tf.Session()
    policy  = create_pi(session,\
                      pick_data,\
                      place_data,\
                      weight_dir,\
                      eval_dir,\
                      args.pi,\
                      args.Qloss,\
                      float(args.d_lr),\
                      float(args.g_lr),\
                      float(args.tau),\
                      int(args.n_score),\
                      float(args.explr_const))
    self.assertEqual( args.tau , policy.pick_pi.tau)
    self.assertEqual( args.tau , policy.place_pi.tau)
    self.assertEqual( args.explr_const , policy.pick_pi.explr_const)
    self.assertEqual( args.explr_const , policy.place_pi.explr_const)
    self.assertEqual( args.d_lr , policy.pick_pi.d_lr)
    self.assertEqual( args.d_lr , policy.place_pi.d_lr)
    self.assertEqual( args.g_lr , policy.pick_pi.g_lr)
    self.assertEqual( args.g_lr , policy.place_pi.g_lr)

if __name__ == '__main__':
  unittest.main()
  
