import subprocess
import numpy as np
import sys


def get_max(algo, n_trial,n_data ):
  out = subprocess.Popen(['./'+algo+'_check_results.sh',str(n_data),str(n_trial)],stdout=subprocess.PIPE)
  stdout = out.communicate()
  results = stdout[0]
  max_score = -np.inf
  max_epoch = 0
  best_epoch = 0

  for line in results.split('\n'):
    if line=='': continue
    mean_score = float(line.split(',')[1])
    std_score  = float(line.split(',')[2])
    epoch      = int(line.split(',')[0])
    """
    elif len(line.split(','))==2 :
      mean_score = float(line.split(',')[0])
      std_score  = float(line.split(',')[1])
    """
    #if epoch>200:
    #  continue

    if mean_score > max_score:
      max_score = mean_score
      best_epoch = epoch
      std_max_score = std_score

    max_epoch +=1
  #print best_epoch
  return max_score,std_max_score,max_epoch,best_epoch

def main():
  algo = sys.argv[1]
    
  n_data_avgs = []
  n_data_stds = []
  #for n_data in range(100,1000,100):
  n_datas = [100,1000,10000]
  n_datas=[100,500,800,1000,2000,4000,6000,8000,10000]
  n_datas = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
  #n_datas = [1000,3000,5000,7000,10000]
  for n_data in n_datas:
    print 'n_data',n_data
    max_scores = []
    n_trials = [0,1,2,3,4,5,6,7]
    for n_trial in n_trials:
      try:
        max_score,std_score,max_epoch,best_epoch = get_max( algo,n_trial,n_data)
      except:
        print 'skipping ',n_data,n_trial
        continue
      print n_trial,max_score,std_score,max_epoch,best_epoch
      max_scores.append(max_score)
    try:
      print "Max,trial,mean,std", np.max(max_scores),np.argmax(max_scores),np.mean(max_scores),np.std(max_scores)
      n_data_avgs.append(np.mean(max_scores))
    except:
      continue
  print n_data_avgs
  
  
if __name__ =='__main__':
  main()


