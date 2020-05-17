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


def get_max_rwds_wrt_samples(fdir, tau):
    file_dir = fdir + '/tau_' + str(tau) + '/explr_p_0.1/n_score_5/n_trial_-1/train_results'
    print file_dir
    max_rwds_over_seeds = []
    max_progress_over_seeds = []
    for result_file in os.listdir(file_dir):
        max_rwds = []
        max_progresses = []
        max_episode_rwd = -np.inf
        max_episode_progress = -np.inf
        if 'performance.txt' in result_file:
            fopen = open(file_dir + '/' + result_file, 'r')
            rdr = csv.reader(fopen, delimiter=',')
            for row in rdr:
                episode_rwd = float(row[1])
                episode_progress = -float(row[2])

                if episode_rwd > max_episode_rwd:
                    max_episode_rwd = episode_rwd
                if episode_progress > max_episode_progress:
                    max_episode_progress = episode_progress

                max_rwds.append(max_episode_rwd)
                max_progresses.append(max_episode_progress)

            max_rwds_over_seeds.append(max_rwds)
            max_progress_over_seeds.append(max_progresses)

    return np.array(max_rwds_over_seeds), np.array(max_progress_over_seeds)


def plot_over_different_taus():
    algo_dir = 'RL_results/convbelt/n_data_100/ppo/dg_lr_0.001_0.0001/'
    tau = 0.3
    rwds, progress = get_max_rwds_wrt_samples(algo_dir, tau)
    avg_rwds = rwds.mean(axis=0)
    rwds_ci = rwds.std(axis=0) * 1.95/np.sqrt(len(rwds))
    print "%.4f %.4f %.4f"% (avg_rwds[499], avg_rwds[999], avg_rwds[1998])
    print "%.4f %.4f %.4f"%(rwds_ci[499], rwds_ci[999], rwds_ci[1998])
    import pdb;pdb.set_trace()
        #sns.tsplot(results, condition=taudir, ci=95, color=np.random.rand(3))

    # best taus:
    #   convbelt, ddpg - 1e-4
    #   mdr, ddpg - 1e-3
    #   mdr, ppo - 0.1
    #   convbelt, ppo - 0.3

def main():
    plot_over_different_taus()


if __name__ == '__main__':
    main()
