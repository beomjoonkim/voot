import pickle
import os
import numpy as np
import sys

all_results = []
uct = sys.argv[1]
widening = sys.argv[2]
alg = sys.argv[3]
planner = sys.argv[4]
if planner=='voronoi_mcts':
    datadir = './voronoi_mcts_results/uct_' + str(uct) + '_widening_' + str(widening) + '_' + alg + '/'
else:
    datadir = './mcts_results/uct_' + str(uct) + '_widening_' + str(widening) + '_' + alg + '/'

avg = 0
n = 0
times = []
n_failed=0
for fin in os.listdir(datadir):
    if fin.find('.pkl') == -1:
        continue
    rwd_time = np.array(pickle.load(open(datadir + fin, 'r')))
    rwd_time = np.array(rwd_time.tolist()['search_time'])

    # print fin,max(rwd_time[:, 1]), max(rwd_time[:,0])

    if max(rwd_time[:, 1]) == 4:
        times.append(max(rwd_time[:, 0]))
    else:
        n_failed +=1

    #if len(times) == 390:
    #    break

times = np.sort(times)
print 'N tried',len(times)
print 'Average success time ', np.mean(times), np.std(times)
print 'Success rate', (len(times) - float(n_failed))/float(len(times))
