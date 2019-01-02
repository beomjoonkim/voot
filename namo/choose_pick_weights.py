import sys
import os
import pickle
import time
import tensorflow as tf
import numpy as np
from PickSOAP import PickSOAP
from PickEvaluator import PickEvaluator

from data_load_utils import load_pick_data
from utils import determine_best_weight_path_for_given_n_data,get_best_weight_file

  

def choose_pick_weights(train_results_dir,n_data,n_trial):
  #if os.path.isfile(train_results_dir+'/wfile_scores.txt'):
  #  print "already evaluated this folder, passing"
  #  return
 
  
  sess    = tf.Session()
  parent_dir = './pick_soap_new_data/'
  data,scalers = load_pick_data( parent_dir=parent_dir,\
                          proc_train_data_dir='./processed_train_data/',\
                          n_data=n_data,n_trial=n_trial)

  x_data = data['x']; c_data = data['c']; s_data = data['s']*1
  c0_data = data['c0']
  opose_data = data['opose'];
  oshape_data = data['oshape']
  x_scaler = scalers['x_scaler']
  w_data = np.hstack([c0_data,opose_data,oshape_data])

  dim_x   = np.shape(x_data)[1]                         # data shape
  dim_w   = np.shape(w_data)[1]                         # context vector shape
  dim_c   = (np.shape(c_data)[1],np.shape(c_data)[-1])   # collision vector shape

  print "N_data = ",x_data.shape

  # load SOAP
  session = tf.Session()
  soap = PickSOAP(session,dim_x,dim_w,dim_c,save_folder=train_results_dir,x_scaler=x_scaler)  
  
  # load the regressor
  path_to_best_evaluator_weight = determine_best_weight_path_for_given_n_data('./pick_evaluator_new_data/',n_data)
  evaluator = PickEvaluator(session,dim_x,dim_w,dim_c)
  evaluator.disc.load_weights(path_to_best_evaluator_weight)
  
  # load the data
  best_score = 0
  score_list = []
  wfile_list = []
  epoch_limit =  np.inf
  print "Using ",path_to_best_evaluator_weight
  for wfile in os.listdir(train_results_dir):
    if wfile.find('a_genepoch') == -1: continue
    epoch = int(wfile.split('_')[2])
    total_epochs = int(wfile.split('_')[3])
    #if total_epochs != 3000 or epoch > epoch_limit: continue
    # generate data
    try:
      soap.a_gen.load_weights( train_results_dir+wfile)
    except:
      print wfile
      continue
    zvals = np.random.normal(size=(n_data,soap.dim_z)).astype('float32')
    Gpred = soap.a_gen.predict( [zvals,w_data,c_data] )

    # evaluate the generated data
    Epred = evaluator.disc.predict( [Gpred,w_data,c_data] )
    eval_score = np.mean(Epred)

    if eval_score>best_score:
      best_score=eval_score
      best_weight=wfile
      f = open(train_results_dir+'/best_weight.txt','w')
      f.write(str(best_score)+'\n')
      f.write(str(best_weight)+'\n')
      f.close()
    score_list.append( eval_score)
    wfile_list.append( wfile)
    print train_results_dir+wfile,eval_score
  ordered_wfile_list = np.array(wfile_list)[ np.argsort(score_list) ].tolist()
  ordered_scores = np.sort(score_list)
  f = open(train_results_dir+'/wfile_scores.txt','w')
  for wfile,score in zip(ordered_wfile_list,ordered_scores): 
    f.write(wfile+','+str(score)+'\n')
  print np.asarray(wfile_list)[np.argsort(score_list)]
  print best_score,best_weight


def main():
  n_data = int(sys.argv[1])
  data_dir = './pick_soap_new_data/n_data_'+str(n_data)+'/'
  for fpath in os.listdir(data_dir):
    n_trial = fpath.split('n_trial_')[1]
    train_results_dir = data_dir+fpath+'/train_results/'
    print train_results_dir
    choose_pick_weights(train_results_dir,n_data,n_trial)
 
if __name__ == '__main__':
  main()


  
