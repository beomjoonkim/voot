import subprocess
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def get_param_configs(algo):
  if algo == 'trpo' or algo == 'gail':
    taus = [0.1,0.2,0.3,0.4,0.5]
    explr_consts = [0.5]
  elif algo == 'soap':
    taus = [0.1,0.5,0.8,1.0,1.2,1.5,2.0,2.2,2.5,3.0]
    taus = [1.0,1.5,2.0,2.5,3.0]
    explr_consts = [0.0]
  elif algo == 'ddpg':
    taus = [1e-1,1e-2,1e-3,1e-4]
    explr_consts = [0.5]
  return taus,explr_consts
def main():
  algos = ['trpo','ddpg','soap','gail']
    
  configs= []
  n_data = 800

  algo_max_scores = {}
  algo_avg_scores = {}
  algo_std_scores = {}
  print 'n_data',n_data
  for algo in algos:
    avgs = []
    stds = []
    maxes = []
    taus,explr_consts = get_param_configs( algo )
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
    algo_max_scores[algo] = max(maxes)
    algo_avg_scores[algo] = max(avgs)
  width = 0.35
  avgs = np.array(avgs)
  stds = np.array(stds)
  p2=plt.bar( range(len(algos)),algo_max_scores.values(),width)
  p1=plt.bar( range(len(algos)),algo_avg_scores.values(),width,bottom=algo_max_scores.values() )
  plt.xticks( range(len(algos)),algo_max_scores.keys() )
  plt.ylabel( 'Rewards' )
  plt.legend((p1[0], p2[0]), ('Avgs', 'Maxes'))
  plt.savefig('./plotters/'+'max_rwds.png')

if __name__ =='__main__':
  main()


