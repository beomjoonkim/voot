import subprocess
import numpy as np
import sys


def get_max( algo,n_trial,n_data,tau,explr_const ):
  out = subprocess.Popen(['./'+algo+'_check_results.sh',str(n_data),str(tau),str(explr_const),str(n_trial)],stdout=subprocess.PIPE)
  stdout = out.communicate()
  results = stdout[0]
  max_score = -np.inf
  max_epoch=0
  epoch = 0
  std_max_score = 0
  for line in results.split('\n'):
    if len(line.split(','))==3:
      mean_score = float(line.split(',')[1])
      std_score  = float(line.split(',')[2])
      epoch      = int(line.split(',')[0])
    elif len(line.split(','))==2 :
      mean_score = float(line.split(',')[0])
      std_score  = float(line.split(',')[1])
      epoch += 1
    else:
      continue
    if mean_score > max_score:
      max_score = mean_score
      std_max_score = std_score
      max_epoch = epoch
  return max_score,std_max_score,max_epoch

def main():
  algo = sys.argv[1]

  #taus = [float(a) for a in sys.argv[2].split(',')]
  #explr_consts = [float(a) for a in sys.argv[3].split(',')]

  if algo == 'trpo':
    taus = [0.3]
    explr_consts = [0.5]
  elif algo == 'gail':
    taus = [0.2]
    explr_consts = [0.5]
  elif algo == 'ddpg':
    taus = [0.001]
    explr_consts = [0.5]
  elif algo == 'soap':
    taus = [2.0]
    explr_consts = [0.0]
  elif algo == 'adq':
    taus = [2.0]
    explr_consts = [0.5]
    
  n_data_avgs = []
  n_data_stds = []
  #for n_data in range(100,1000,100):
  n_datas = range(100,1000,100)
  n_datas.append(1500)
  n_datas.append(2000)
  n_datas.append(3000)
  n_datas.append(4000)
  n_datas.append(5000)
  best_score=-np.inf

  for n_data in n_datas:
    print 'n_data',n_data
    for explr_const in explr_consts:
      for tau in taus:
        max_scores = []
        for n_trial in [0,1,2,3]:
          max_score,std_score,max_epoch= get_max( algo,n_trial,n_data,tau,explr_const )
          print n_trial,max_score,std_score,max_epoch
          if max_score > best_score:
            best_trial = n_trial
            best_epoch = max_epoch
            best_n_data = n_data
            best_score = max_score
          max_scores.append(max_score)
        try:
          print "Max, mean, std", np.max(max_scores),np.mean(max_scores),np.std(max_scores)
        except:
          continue
  print "n_data,score,trial, epoch", best_n_data,best_score,best_trial,best_epoch

if __name__ =='__main__':
  main()


