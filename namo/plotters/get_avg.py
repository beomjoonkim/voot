import sys
import subprocess
import numpy as np
def get_avg( performance_file ):
  #out = subprocess.Popen(['./'+algo+'_check_results.sh',str(n_data),str(n_trial)],stdout=subprocess.PIPE)
  results = open( performance_file,'r').readlines()
  max_score = -np.inf
  max_epoch = 0
  best_epoch = 0

  avgs = []
  for line in results:
    if len(line.split(','))==3:
      mean_score = float(line.split(',')[1])
      std_score  = float(line.split(',')[2])
    elif len(line.split(','))==2 :
      mean_score = float(line.split(',')[0])
      std_score  = float(line.split(',')[1])
    else:
      continue
    avgs.append(mean_score)
  #print best_epoch
  print len(avgs)
  return np.mean(np.sort(avgs)[-30:]),np.std(avgs)

def main():
  algo = sys.argv[1]
  for n_data in range(1000,11000,1000):
    try:
      if algo.find('soap')!=-1:
        performance_file = '/data/public/rw/pass.port/NAMO/n_data_'+str(n_data)+'/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_fixed_halfpick/eval_results/performance_with_noise.txt'
        #performance_file = "/data/public/rw/pass.port/NAMO/n_data_"+str(n_data)+"/soap/adv/tau_2.0/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/n_trial_-1/eval_results/performance_with_noise.txt"
      elif algo.find('trpo')!=-1:
        performance_file = '/data/public/rw/pass.port/NAMO/n_data_'+str(n_data)+'/trpo/adv/tau_0.3/dg_lr_0.0001_0.0001/n_score_1/architecture_0/explr_const_0.5/performance_with_noise.txt'
      print n_data,get_avg(performance_file)
    except:
      continue

if __name__ == '__main__':
  main()
