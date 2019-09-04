import os
import numpy as np
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler


def one_hot_encode(vec):
    n_elements = len(np.unique(vec))
    n_data = np.shape(vec)[0]
    encoded = np.zeros((n_data, n_elements))
    for i in range(n_data):
        encoded[i, vec[i, :]] = 1
    return encoded


def create_bit_encoding_of_konf(n_konf):
    n_helper = 10
    k_data = np.ones((n_konf, n_helper)) * -1
    idnumb = 1
    for idx in range(n_konf):
        # binstr=bin(idnumb)[2:]
        binstr = '{0:010b}'.format(idnumb)
        binidx = range(len(binstr))[::-1]
        for k in binidx:
            if int(binstr[k]) == 1:
                k_data[idx, k] = 1
            else:
                k_data[idx, k] = -1
        idnumb += 1
    k_data = k_data.reshape((n_konf, n_helper))
    return k_data


def convert_collision_vec_to_one_hot(c_data):
    n_konf = c_data.shape[1]
    onehot_cdata = []
    for cvec in c_data:
        one_hot_cvec = np.zeros((n_konf, 2))
        for boolean_collision, onehot_collision in zip(cvec, one_hot_cvec):
            onehot_collision[boolean_collision] = 1
        assert (np.all(np.sum(one_hot_cvec, axis=1) == 1))
        onehot_cdata.append(one_hot_cvec)

    """
    # test code
    for onehot_vec,cvec in zip(onehot_cdata,c_data):
      for onehot_collision,boolean_collision in zip(onehot_vec,cvec):
        assert( onehot_collision[boolean_collision] == 1 )
        assert( np.sum(onehot_collision) == 1 )
    """
    onehot_cdata = np.array(onehot_cdata)
    return onehot_cdata


def aggregate_data(train_dir, fname_keyword):
    # loops through train files in train dir, loading data from files with fname_keyword in it
    train_dir = train_dir + '/'
    n_episodes = 0
    for fdata in os.listdir(train_dir):
        if fdata.find(fname_keyword) == -1: continue
        if fdata.find('aggregated') != -1: continue
        if fdata.find('box') != -1: continue
        data = pickle.load(open(train_dir + fdata))

        if 'x_data' not in locals():
            x_data = np.array(data[0])
            s_data = np.array(data[1])[:, None]
            c_data = np.array(data[2])
            oidx_data = np.array(data[3])[:, None]
        else:
            x_data = np.vstack((x_data, np.array(data[0])))
            s_data = np.vstack((s_data, np.array(data[1])[:, None]))
            c_data = np.vstack((c_data, np.array(data[2])))
            oidx_data = np.vstack((oidx_data, np.array(data[3])[:, None]))
        n_episodes += 1
        data = [x_data, s_data, c_data, oidx_data]
        if len(x_data) > 5000:
            # print 'saving',len(x_data),n_episodes
            with open(train_dir + fname_keyword + '_aggregated_data.p', 'wb') as foutput:
                pickle.dump(data, foutput)
            return





def setup_planner_result_dir(parent_dir, args):
    alg = args.pi
    if alg == 'unif':
        return parent_dir + '/unif/'

    n_data = args.n_data
    n_trial = args.n_trial
    d_lr = args.d_lr
    g_lr = args.g_lr
    tau = args.tau  # epsilon in TRPO, tau in DDPG, lambda in SOAP
    v = args.v
    pi_name = args.pi
    explr_const = args.explr_const
    nscore_train = args.n_score

    n_data_dir = parent_dir + '/n_data_' + str(n_data)
    pi_dir = n_data_dir + '/' + pi_name
    lr_dir = pi_dir + '/dg_lr_' + str(d_lr) + '_' + str(g_lr) + '/'
    tau_dir = lr_dir + '/tau_' + str(tau) + '/'
    explr_const_dir = tau_dir + '/' + '/explr_const_' + str(explr_const) + '/'
    nscore_dir = explr_const_dir + '/' + '/n_score_' + str(nscore_train) + '/'
    trial_dir = nscore_dir + '/n_trial_' + str(n_trial)

    plan_dir = trial_dir + '/planner_result/epoch_' + str(args.epoch)

    if not os.path.exists(plan_dir):
        os.makedirs(plan_dir)

    return plan_dir


def get_planning_trajectory_data(traj_f):
    data = pickle.load(open(traj_f))
    rewards = data[1]
    actions = data[0]
    states = convert_collision_vec_to_one_hot(np.array(data[2]) * 1)
    return states, actions, rewards


def format_RL_data(trajs, n_data=None):
    # this function creates state action and sum of rewards of each tracj
    # R(\tau),s,a
    traj_lengths = []
    n_trans = 0
    for tau in trajs:
        if len(tau['a']) == 0:
            continue
        idx = 0
        for s, a, r in zip(tau['s'], tau['a'], tau['r']):
            traj_lengths.append(len(tau['s']) - idx)
            idx += 1
            a = np.array(a.continuous_parameters['action_parameters'])
            a = a.reshape((len(a),))
            """
            if a[2] > np.pi * 2:
                a[2] -= 2 * np.pi
            if a[2] < 0:
                a[2] += 2 * np.pi
            """

            if 'states' in locals():
                states = np.r_[states, s]  # How do I mark a terminal state?
                actions = np.r_[actions, a[None, :]]
                rewards = np.r_[rewards, [r]]
            else:
                states = s
                actions = a[None, :]
                rewards = [r]
        if len(states) == 0:
            continue
        if len(states) == 1:
            traj_sprime = np.zeros(states.shape)
        else:
            traj_sprime = np.array(states)[1:, :]
            traj_sprime = np.r_[traj_sprime, np.zeros(traj_sprime[0, :].shape)[None, :]]  # terminal state

        traj_states = np.array(states)
        traj_actions = np.array(actions)
        traj_rewards = np.array(rewards)
        traj_sum_rwds = [np.sum(rewards) - i for i in range(len(traj_rewards))]
        traj_scores = [np.sum(rewards) for i in range(len(traj_rewards))]

        if 'all_states' in locals():
            all_states = np.r_[all_states, traj_states]
            all_sprime = np.r_[all_sprime, traj_sprime]
            all_actions = np.r_[all_actions, traj_actions]
            all_rewards = np.r_[all_rewards, traj_rewards]
            all_sum_rewards = np.r_[all_sum_rewards, traj_sum_rwds]
            all_scores = np.r_[all_scores, traj_scores]
        else:
            all_states = traj_states
            all_sprime = traj_sprime
            all_actions = traj_actions
            all_rewards = traj_rewards
            all_sum_rewards = traj_sum_rwds
            all_scores = traj_scores
        del states, actions, rewards
        if n_data is not None and len(all_states) >= n_data: break
    if 'all_states' not in locals():
        return np.array([]), np.array([]), \
               np.array([]), np.array([]), \
               np.array([]), np.array([]), traj_lengths
    else:
        n_konf = max(all_states.shape[1:])
        n_data = len(all_actions)
        dim_state = all_states.shape[1]
        dim_actions = all_actions.shape[1]
        all_states = np.array(all_states)
        all_states = all_states.reshape((n_data, dim_state))
        all_actions = np.array(all_actions)
        all_actions = all_actions.reshape((n_data, dim_actions))
        all_rewards = np.array(all_rewards)
        all_rewards = all_rewards.reshape((n_data, 1))
        all_sprime = np.array(all_sprime)
        all_sprime = all_sprime.reshape((n_data, dim_state))
        all_sum_rewards = np.array(all_sum_rewards)
        all_sum_rewards = all_sum_rewards.reshape((n_data, 1))
        all_scores = np.array(all_scores)

        """
        return np.array(all_states),np.array(all_actions),\
               np.array(all_rewards),np.array(all_sprime),\
               np.array(all_sum_rewards),np.array(all_scores),traj_lengths
        """
        return all_states, all_actions, all_rewards, all_sprime, all_sum_rewards, all_scores, traj_lengths


def load_place_RL_data(parent_dir, n_data):
    traj_data = []

    import time
    stime = time.time()
    n_trans = 0
    max_sumRs = []
    for fdata in os.listdir('./processed_train_data/'):
        if fdata.find('aggregated') != -1: continue
        if fdata.find('box') != -1: continue
        if fdata.find("RL_train_data_") == -1: continue  # preprocess_place_data makes this file
        trajs_from_episode = pickle.load(open('./processed_train_data/' + fdata, 'r'))

        max_sumR = 0
        for traj in trajs_from_episode:
            cvecs = np.array(traj['s']) * 1
            if len(cvecs.shape) == 1: cvecs = cvecs[None, :]
            if (cvecs.shape[1]) == 0: continue
            state = convert_collision_vec_to_one_hot(cvecs)
            state = np.tile(state, (1, 1, 1))
            state = state[:, :, :, None]
            traj['s'] = [s[None, :] for s in state]
            traj['a'] = np.array(traj['a'])[:, 3:]  # take robot base pose
            traj_data.append(traj)
            n_trans += len(cvecs)
            if max_sumR < np.sum(traj['r']):
                max_sumR = np.sum(traj['r'])
        max_sumRs.append(max_sumR)
        print np.mean(max_sumRs)
        if n_trans >= n_data:
            break

    S, A, R, Sprime, sumR, scores, traj_lengths = format_RL_data(traj_data, n_data)
    data = [S[:n_data, :], A[:n_data, :], R[:n_data], Sprime[:n_data, :], sumR[:n_data], scores, traj_lengths]
    return data


def load_key_configs():
    key_configs = np.array(pickle.load(open('./key_configs/key_configs.p', 'r')))
    return key_configs


def get_success_traj(traj_f):
    data = pickle.load(open(traj_f))
    scores = data[1]
    actions = np.array(data[0])
    states = convert_collision_vec_to_one_hot(np.array(data[2]) * 1)
    return states[scores >= 5, :], actions[scores >= 5, :]

    n_data_dir = parent_dir + '/n_data_' + str(n_data)

    trial_dir = n_data_dir + '/n_trial_' + str(n_trial)
    scaler_dir = trial_dir + '/scalers/'
    train_results_dir = trial_dir + '/train_results/'

    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    if not os.path.exists(n_data_dir):
        os.mkdir(n_data_dir)
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)
    if not os.path.exists(scaler_dir):
        os.mkdir(scaler_dir)
    if not os.path.exists(train_results_dir):
        os.mkdir(train_results_dir)
    return scaler_dir


def load_guidance_data(parent_dir, proc_train_data_dir, n_data, n_trial):
    # returns a list of (s,a) pairs that succeded
    setup_save_dirs(parent_dir, n_data, n_trial)
    trajs = []
    for fdata in os.listdir(proc_train_data_dir):
        if fdata.find('aggregated') != -1: continue
        if fdata.find('box') != -1: continue
        if fdata.find("place_train_data_") == -1: continue
        s, a = get_success_traj(proc_train_data_dir + '/' + fdata)
        trajs.append({'s': s, 'a': a})
        if len(trajs) > 199:
            break

    for tau in trajs:
        for s, a in zip(tau['s'], tau['a']):
            if 'states' in locals():
                states = np.r_[states, s]
                actions = np.r_[actions, a]
            else:
                states = s
                actions = a
    states = states[:, :, :, None]
    return np.array(states), np.array(actions)


def load_place_data(parent_dir, proc_train_data_dir, n_data, n_trial):
    # normalizes and returns the data for SOAP
    scaler_dir = setup_save_dirs(parent_dir, n_data, n_trial)

    if not os.path.exists(proc_train_data_dir + '/place_aggregated_data.p'):
        aggregate_data(proc_train_data_dir, 'place')
    data = pickle.load(open(proc_train_data_dir + '/place_aggregated_data.p', 'r'))

    # load score data
    s_data = np.array(data[1])[:n_data, :]  # scores

    # x data; predict robot pose, and make sure angles are between 0 to 2pi
    SCALE_SWITCH = True
    # x_data = np.array(data[0])[:n_data,:3] # actions - predict obj pose
    x_data = np.array(data[0])[:n_data, :]  # actions - predict entire pick action
    x_scaler = StandardScaler(with_mean=SCALE_SWITCH, with_std=SCALE_SWITCH)
    scaled_x = x_scaler.fit_transform(x_data)
    c_data = np.array(data[2])[:n_data, :] * 1  # c vectors

    # load konfs and select the subset for which c_data is not zero
    key_config_idxs = np.where(np.sum(c_data, axis=0) != 0)[0]
    pickle.dump(key_config_idxs, open('./key_configs/key_config_idxs_n_' + str(n_data) + '.p', 'wb'))
    key_configs = np.array(pickle.load(open('./key_configs/key_configs.p', 'r')))
    key_configs = key_configs[key_config_idxs, :]

    n_konf = len(key_config_idxs)
    dim_konf = key_configs.shape[1]

    # make k to encode binary identification
    k_data = key_configs
    k_scaler = StandardScaler(with_mean=SCALE_SWITCH, with_std=SCALE_SWITCH)
    scaled_k = k_scaler.fit_transform(k_data)
    scaled_k = np.tile(scaled_k, (n_data, 1, 1))  # shape = n_data,n_conf,dim_conf
    scaled_k = scaled_k.reshape((n_data, n_konf, 3, 1))

    # filter the c vector
    c_data = c_data[:, key_config_idxs]
    c_data = c_data * 1
    c_data = convert_collision_vec_to_one_hot(c_data)
    c_scaler = StandardScaler(with_mean=False, with_std=False)
    scaled_c = c_data
    scaled_c = scaled_c.reshape((n_data, n_konf, 2))

    # save the scalers
    pickle.dump(x_scaler, open(scaler_dir + '/x_scaler.p', 'wb'))
    pickle.dump(k_scaler, open(scaler_dir + '/k_scaler.p', 'wb'))
    pickle.dump(c_scaler, open(scaler_dir + '/c_scaler.p', 'wb'))

    assert (np.all(np.isclose(x_scaler.inverse_transform(scaled_x), x_data)))

    # return the scaled data
    data = {'x': scaled_x, 'c': scaled_c, 'k': scaled_k, 's': s_data, \
            'key_configs': key_configs, \
            'x_scaler': x_scaler, 'c_scaler': c_scaler, 'k_scaler': k_scaler}
    return data


def main():
    data = load_place_data('./place_evaluator/', './processed_train_data/', n_data=5000, n_trial=0)


if __name__ == '__main__':
    main()
