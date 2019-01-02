import pickle
import argparse
import os
import numpy as np

def get_stripstream_results(domain_name):
    if domain_name == 'convbelt':
        result_dir = './test_results/convbelt_results/stripstream/'
    search_times = []
    success = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1: 
            continue
        try:
          result = pickle.load(open(result_dir+fin,'r'))
        except:
          print fin
        success.append(result['plan'] is not None)
        search_times.append(result['search_time'])
    print "stripstream time and success rate:"
    print np.array(search_times).mean()
    print np.array(success).mean()
    print len(search_times)

def get_mcts_results(domain_name):
    if domain_name == 'convbelt':
        result_dir = './test_results/convbelt_results/uct_0.0_widening_0.5_unif/'
    search_times = []
    success = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1: 
            continue
        try:
          result = pickle.load(open(result_dir+fin,'r'))
        except:
          print fin
        success.append(result['plan'] is not None)
        search_times.append(result['search_time'][-1][0])
        #if result['plan'] is None:
        #    import pdb;pdb.set_trace()

    print "mcts time and success rate:"
    print np.array(search_times).mean()
    print np.array(success).mean()
    print len(search_times)
    

def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-planner', type=str, default='stripstream')
    args = parser.parse_args()
    get_stripstream_results(args.domain)
    get_mcts_results(args.domain)


if __name__ == '__main__':
    main()
