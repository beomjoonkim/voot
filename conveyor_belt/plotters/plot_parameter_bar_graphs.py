import subprocess
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_max_bar_graphs import get_param_configs

def get_max( algo,n_trial,n_data,tau,explr_const ):
  out = subprocess.Popen(['./'+algo+'_check_results.sh',str(n_data),str(tau),str(explr_const),str(n_trial)],stdout=subprocess.PIPE)
  stdout = out.communicate()
  results = stdout[0]
  max_score = -np.inf
  for line in results.split('\n'):
    if len(line.split(','))==3:
      mean_score = float(line.split(',')[1])
      std_score  = float(line.split(',')[2])
      epoch      = int(line.split(',')[0])
    elif len(line.split(','))==2 :
      mean_score = float(line.split(',')[0])
      std_score  = float(line.split(',')[1])
    else:
      continue
    if mean_score > max_score:
        max_score = mean_score
        std_max_score = std_score
  return max_score,std_score

def get_xlabel(algo):
  if algo == 'trpo' or algo == 'gail':
    xlabel = 'Clipping values'
  elif algo == 'soap':
    xlabel = 'Adversarial parameter'
  elif algo == 'ddpg':
    xlabel = 'Soft update parameter'
  return xlabel

def main():
  #algos = ['trpo','ddpg','soap','gail']
  algos = ['trpo','ddpg','soap','gail']
  n_data = 800

  for algo in algos:
    taus,explr_consts = get_param_configs( algo )
    avgs = []
    stds = []
    maxes = []
    configs= []

    for explr_const in explr_consts:
      for tau in taus:
        print 'tau,explr_const',tau,explr_const
        max_scores = []
        for n_trial in [0,1,2,3]:
          try:
            max_score,std_score= get_max( algo,n_trial,n_data,tau,explr_const )
          except:
            continue
          print max_score,std_score
          max_scores.append(max_score)
        print "Mean, std", np.mean(max_scores),np.std(max_scores)
        avgs.append( np.mean(max_scores) )
        stds.append( np.std(max_scores) )
        maxes.append( np.max(max_scores) )
        configs.append(str(tau))
    width = 0.35
    avgs = np.array(avgs)
    stds = np.array(stds)
    plt.figure()
    p2=plt.bar( range(len(configs)),maxes,width)
    p1=plt.bar( range(len(configs)),avgs,width, yerr = 1.96*np.array(stds)/2.0,bottom=maxes )
    plt.xticks( range(len(configs)),configs )
    plt.ylabel( 'Rewards' )
    xlabel=get_xlabel(algo)
    plt.xlabel(xlabel)
    plt.legend((p1[0], p2[0]), ('Avgs', 'Maxes'))
    plt.savefig('./plotters/'+algo+'_rwd_config.png')
    plt.close('all')

if __name__ =='__main__':
  main()


