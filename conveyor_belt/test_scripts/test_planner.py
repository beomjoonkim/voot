import sys
import os
import argparse
import socket
import pickle
import tensorflow as tf

from conveyor_belt_env import ConveyorBelt
from planners.forward_search import forward_search

from generators.PlaceSOAP import PlaceSOAP
from generators.PlaceDDPG import PlaceDDPG
from generators.PlaceTRPO import PlaceTRPO
from generators.PlaceUniform import PlaceUnif
from generators.PlaceGAIL import PlaceGAIL

from openravepy import *

SOAP_WPATH = "/data/public/rw/pass.port/conveyor_belt/n_data_4000/soap/dg_lr_0.001_0.0001/tau_2.0/explr_const_0.0/n_score_5/n_trial_2/train_results/a_gen_epoch_69.h5"
SOAP_RESULT = '/data/public/rw/pass.port/conveyor_belt/n_data_4000/soap/dg_lr_0.001_0.0001/tau_2.0/explr_const_0.0/n_score_5/n_trial_2/planner_result/'
SOAP_WPATH = "./a_gen_epoch_69.h5"

TRPO_WPATH = "/data/public/rw/pass.port/conveyor_belt/n_data_5000/trpo///dg_lr_0.001_0.0001/tau_0.3/explr_const_0.5/n_score_5/n_trial_0/train_results/a_genepoch_163_2.6.h5"
TRPO_WPATH = "/data/public/rw/pass.port/conveyor_belt/n_data_5000/trpo///dg_lr_0.001_0.0001/tau_0.3/explr_const_0.5/n_score_5/n_trial_0/planner_result/"

GAIL_WPATH = "/data/public/rw/pass.port/conveyor_belt/n_data_5000/gail/dg_lr_0.001_0.0001/tau_0.2/explr_const_0.5/n_score_5/n_trial_0/train_results/a_genepoch_58_2.6.h5"
GAIL_RESULT = "/data/public/rw/pass.port/conveyor_belt/n_data_5000/gail/dg_lr_0.001_0.0001/tau_0.2/explr_const_0.5/n_score_5/n_trial_0/planner_result/"

DDPG_WPATH = "/data/public/rw/pass.port/conveyor_belt/n_data_5000/ddpg_new//dg_lr_0.001_0.0001/tau_0.001/explr_const_0.5/n_score_5/n_trial_2/train_results/a_gentau_0.001epoch_3_2.2.h5"
DDPG_WPATH = "/data/public/rw/pass.port/conveyor_belt/n_data_5000/ddpg_new//dg_lr_0.001_0.0001/tau_0.001/explr_const_0.5/n_score_5/n_trial_2/planner_result/"

if socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = '../../AdvActorCriticConveyorBeltResults/'
else:
    ROOTDIR = '/data/public/rw/pass.port//conveyor_belt/'


def create_policy(args, problem):
    key_configs = pickle.load(open('./key_configs/key_configs.p', 'r'))
    dim_state = (957, 2)
    dim_action = 3

    alg = args.pi
    explr_const = args.explr_const

    v = args.v
    session = tf.Session()

    if alg == 'soap':
        x_scaler = pickle.load(open('./x_scaler.pkl', 'r'))
        policy = PlaceSOAP(session,
                           dim_action,
                           dim_state,
                           key_configs=key_configs,
                           x_scaler=x_scaler,
                           tau=2.0,
                           save_folder="",
                           explr_const=explr_const,
                           visualize=v)
    elif alg.find('ddpg') != -1:
        x_scaler = pickle.load(
            open('/data/public/rw/pass.port/conveyor_belt/n_data_5000/ddpg/scalers/x_scaler.pkl', 'r'))
        policy = PlaceDDPG(session,
                           dim_action,
                           dim_state,
                           key_configs=key_configs,
                           x_scaler=x_scaler,
                           tau=0.001,
                           save_folder="",
                           explr_const=0.5,
                           visualize=v)
    elif alg == 'trpo':
        x_scaler = pickle.load(
            open('/data/public/rw/pass.port/conveyor_belt/n_data_5000/trpo/scalers/x_scaler.pkl', 'r'))
        policy = PlaceTRPO(session,
                           dim_action,
                           dim_state,
                           key_configs=key_configs,
                           tau=0.3,
                           x_scaler=x_scaler,
                           save_folder="",
                           explr_const=0.5,
                           visualize=v)
    elif alg == 'gail':
        x_scaler = pickle.load(
            open('/data/public/rw/pass.port/conveyor_belt/n_data_5000/gail/scalers/x_scaler.pkl', 'r'))
        policy = PlaceGAIL(session,
                           dim_action,
                           dim_state,
                           key_configs=key_configs,
                           a_scaler=x_scaler,
                           tau=0.2,
                           save_folder="",
                           explr_const=0.5,
                           visualize=v)
    elif alg == 'unif':
        policy = PlaceUnif(problem['env'], problem['env'].GetRobots()[0], problem['loading_region'],
                           problem['all_region'])

    return policy


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-n_episode', type=int, default=0)
    parser.add_argument('-planner', default='forward_search')
    parser.add_argument('-pi', default='unif')
    parser.add_argument('-tau', type=float, default=1e-5)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-n_score', type=int, default=5)
    parser.add_argument('-epoch', type=int, default=0)
    parser.add_argument('-explr_const', type=float, default=0.0)
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-n_trial', type=int, default=-1)
    args = parser.parse_args()
    return args


def get_weight_file_name(epoch, train_results_dir):
    for wfile in os.listdir(train_results_dir):
        if wfile.find('epoch_' + str(epoch) + "_") != -1 or \
                wfile.find('epoch_' + str(epoch) + ".h5") != -1:
            if wfile.find('a_gen') != -1:
                agen = wfile
            elif wfile.find('disc') != -1:
                disc = wfile
    return agen, disc


def main():
    args = parse_args()
    n_episode = args.n_episode
    v = args.v

    convbelt = ConveyorBelt(v)

    pi = create_policy(args, convbelt.problem)
    if args.pi.find('soap') != - 1:
        planner_result_dir = SOAP_RESULT
        pi.a_gen.load_weights(SOAP_WPATH)
    elif args.pi.find('trpo') != -1:
        planner_result_dir = TRPO_RESULT
        pi.a_gen.load_weights(TRPO_WPATH)
    elif args.pi.find('gail') != -1:
        planner_result_dir = GAIL_RESULT
        pi.a_gen.load_weights(GAIL_WPATH)
    elif args.pi.find('ddpg') != -1:
        planner_result_dir = DDPG_RESULT
        pi.a_gen.load_weights(DDPG_WPATH)
    elif args.pi.find('unif') != -1:
        planner_result_dir = '/data/public/rw/pass.port/conveyor_belt/unif_planning/'

    if args.planner == 'forward_search':
        nodes, rwd_time_list = forward_search(convbelt, max_exp=50, policy=pi)

    if not os.path.isdir(planner_result_dir):
        os.makedirs(planner_result_dir)

    nodes_to_save = [n for n in nodes]
    for n in nodes_to_save:
        n.state = n.state[1]

    pickle.dump({'obst_shapes': convbelt.problem['obst_shapes'],
                 'obst_poses': convbelt.problem['obst_poses'],
                 'obj_shapes': convbelt.problem['obj_shapes'],
                 'obj_poses': convbelt.problem['obj_poses'],
                 'nodes': nodes_to_save,
                 'rwd_time_list': rwd_time_list},
                  open(planner_result_dir + 'result_' + str(args.n_trial) + '.pkl', 'wb'))

    convbelt.problem['env'].Destroy()


if __name__ == '__main__':
    main()
