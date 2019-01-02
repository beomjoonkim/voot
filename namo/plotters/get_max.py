import sys
import subprocess
import numpy as np
def get_max( performance_file ):
  #out = subprocess.Popen(['./'+algo+'_check_results.sh',str(n_data),str(n_trial)],stdout=subprocess.PIPE)
  results = open( performance_file,'r').readlines()
  max_score = -np.inf
  max_epoch = 0
  best_epoch = 0
  epoch = 0
  for line in results:
    if len(line.split(','))==3:
      mean_score = float(line.split(',')[1])
      std_score  = float(line.split(',')[2])
      try:
        epoch      = int(line.split(',')[0])
      except:
        epoch =0
    elif len(line.split(','))==2 :
      mean_score = float(line.split(',')[0])
      std_score  = float(line.split(',')[1])
    else:
      continue
    if mean_score > max_score:
      max_score = mean_score
      best_epoch = epoch
      std_max_score = std_score
    max_epoch +=1
    if max_epoch > 200:
      break
  #print best_epoch
  return max_score,std_score,max_epoch,best_epoch

def main():
  performance_file = sys.argv[1]
  print get_max(performance_file)

if __name__ == '__main__':
  main()
