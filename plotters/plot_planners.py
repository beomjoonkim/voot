import pickle
import argparse
import os
import numpy as np


def get_stripstream_results(domain_name):
    if domain_name == 'convbelt':
        result_dir = './test_results/convbelt_results/stripstream/'
    else:
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
        is_success = result['plan'] is not None
        success.append(is_success)
        if is_success:
          search_times.append(result['search_time'])
    print "stripstream time and success rate:"
    print np.array(search_times).mean()
    print np.array(success).mean()
    print len(search_times)


def get_mcts_results(domain_name):
    if domain_name == 'convbelt':
        result_dir = './test_results/convbelt_results/uct_0.0_widening_0.5_unif/'
        result_dir = './test_results/convbelt_results/uct_0.0_widening_0.5_voo/'
    elif domain_name == 'namo':
        result_dir = './test_results/namo_results/uct_0.0_widening_0.5_voo/'
        #result_dir = './test_results/namo_results/uct_0.0_widening_0.5_unif/'
    search_times = []
    success = []
    search_rwd_times = []
    for fin in os.listdir(result_dir):
        if fin.find('.pkl') == -1: 
            continue
        try:
          result = pickle.load(open(result_dir+fin,'r'))
        except:
          print fin

        search_rwd_times.append(result['search_time'])
        if domain_name=='convbelt':
            is_success = result['plan'] is not None
            success.append(is_success)
            if is_success:
                search_times.append(result['search_time'][-1][0])
        else:
            is_success = result['search_time']['namo'][-1][-1]
            success.append(is_success)
            if is_success:
              search_times.append(result['search_time']['namo'][-1][0])

    print "mcts time and success rate:"
    print np.array(search_times).mean()
    print np.array(success).mean()
    print len(success)
    import pdb;pdb.set_trace()
    

def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-planner', type=str, default='stripstream')
    args = parser.parse_args()
    #get_stripstream_results(args.domain)
    get_mcts_results(args.domain)


if __name__ == '__main__':
    main()
