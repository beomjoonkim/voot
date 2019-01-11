import numpy as np
import cPickle as pickle


class BOX:
  def __init__(self, C, infeasible_rwd): 
    self.C = C
    self.infeasible_rwd = infeasible_rwd
  
  def train(self, policy_params, scores, problem_insts ):
    # this function constructs an experience matrix
    exp_mat = []
    total_eval = problem_insts.shape[0] * policy_params.shape[0]
    progress = 0
  
    """
    for w_inst in problem_insts:
      env = BipedalWalker(w_inst[0],w_inst[1],\
                          w_inst[2],w_inst[3],\
                          w_inst[4])
      score_vector = [] 
      for policy in policy_params: 
        g_model = convert_vec_to_model(policy[None,])  
        rwd = run_model(env,g_model,1000,False)
        score_vector.append(rwd)
        progress += 1
        print 'progress = %d/%d'%(progress,total_eval)
      exp_mat.append(score_vector)
    """
    self.exp_mat = np.array(exp_mat)
    self.policy_params = policy_params
    
  def select_arm(self, mu, cov, C ):
    return np.argmax( mu + C*np.diag(cov))

  def update_mu(self, problem_w, mu, cov, eval_idxs ): 
    uneval_idxs =list( (set(range(len(mu)))).difference(set(eval_idxs)) )
    
    eval_diff = np.asmatrix(problem_w[0,eval_idxs] - mu[eval_idxs]).transpose()
    # ix_ function builds cross products of elements of first and second inputs
    cov_uneval_eval = cov[np.ix_(uneval_idxs,eval_idxs)] 
    cov_eval_eval =  cov[np.ix_(eval_idxs,eval_idxs)] 
    reg_term = 0.000001*np.eye( len(eval_idxs) )
    mu_update = np.asmatrix(mu[uneval_idxs]).transpose()\
                 + np.dot(np.dot(cov_uneval_eval,np.linalg.inv(cov_eval_eval+reg_term)),eval_diff)
    mu_update = np.asarray(mu_update).reshape( (mu_update.shape[0],))

    # append evaluated values to updated_mu for evaled_idxs
    temp = np.zeros(np.shape(mu))
    temp[uneval_idxs] = np.asarray(mu_update)
    temp[eval_idxs]   = problem_w[0,eval_idxs]
    mu_update = temp
    
    return mu_update

  def update_cov( self, eval_idxs, cov ):
    uneval_idxs =list( (set(range(len(cov[0,:])))).difference(set(eval_idxs)) )
    cov_eval_uneval = cov[np.ix_(eval_idxs,uneval_idxs)] 
    cov_eval_eval =  cov[np.ix_(eval_idxs,eval_idxs)] 
    cov_uneval_uneval = cov[np.ix_(uneval_idxs,uneval_idxs)] 
    cov_uneval_eval = cov[np.ix_(uneval_idxs,eval_idxs)] 

    reg_term = 0.000001*np.eye( len(eval_idxs) )

    cov_update = cov_uneval_uneval -\
                 np.dot(np.dot(cov_uneval_eval,np.linalg.inv(cov_eval_eval+reg_term)),
                        cov_eval_uneval)

    # append dummy values 
    temp = np.zeros(np.shape(cov))
    temp[np.ix_(uneval_idxs,uneval_idxs)] = cov_update
    cov_update = temp
    
    return cov_update
