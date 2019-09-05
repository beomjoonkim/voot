import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import numpy as np

def savefig(xlabel, ylabel, fname=''):
    plt.legend(loc='best', prop={'size': 13})
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print 'Saving figure ', fname + '.png'
    plt.savefig(fname + '.png', dpi=100, format='png')


def get_results_across_seeds(fdir):
    explr_dirs = os.listdir(fdir)
    for explr_dir in explr_dirs:
        file_dir = fdir+'/'+explr_dir+'/n_score_5/n_trial_-1/train_results'
        max_sumR_over_seeds = []
        for result_file in os.listdir(file_dir):
            max_sumRs = []
            max_sumR = -np.inf
            if 'performance.txt' in result_file:
                fopen = open(file_dir+'/'+result_file, 'r')
                rdr = csv.reader(fopen, delimiter=',')
                for row in rdr:
                    episode_sumR = float(row[1])
                    if episode_sumR > max_sumR:
                        max_sumR = episode_sumR
                    max_sumRs.append(max_sumR)
                max_sumR_over_seeds.append(max_sumRs)
    return np.array(max_sumR_over_seeds)


def plot_across_algorithms():
    algo_dir = 'RL_results/mdr/n_data_100/ddpg/dg_lr_0.001_0.0001/'
    taudirs = os.listdir(algo_dir)

    for taudir in taudirs:
        results = get_results_across_seeds(algo_dir+taudir)
        sns.tsplot(results, condition=taudir,ci=95,color=np.random.rand(3))
    savefig('X', 'Rwds')


if __name__ == '__main__':
    plot_across_algorithms()
