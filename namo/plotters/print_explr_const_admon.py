import subprocess
import numpy as np
import sys

def get_max(n_trial,n_data,explr_const ):
  if float(explr_const) != 0.5:
    out = subprocess.Popen(['./admon_explr_const_check_results.sh',str(n_data),str(explr_const),str(n_trial)],stdout=subprocess.PIPE)
  else:
    out = subprocess.Popen(['./admon_check_results.sh',str(n_data),str(n_trial)],stdout=subprocess.PIPE)
  stdout = out.communicate()
  results = stdout[0]
  max_score = -np.inf
  max_epoch = -np.inf
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
    if epoch > max_epoch:
      max_epoch = epoch
  return max_score,std_score,max_epoch

def main():
  explr_const=sys.argv[1]
  n_datas = [100,500,800,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  n_datas = [500]
  for n_data in n_datas:
    max_scores = []
    for n_trial in [0,1,2,3]:
      max_score,std_score,max_epoch = get_max( n_trial,n_data,explr_const)
      max_scores.append(max_score)
      print max_scores
    print "Max, mean, std", np.max(max_scores),np.mean(max_scores),np.std(max_scores)

if __name__ == '__main__':
    main()

