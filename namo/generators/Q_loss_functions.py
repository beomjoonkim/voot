import sys
import tensorflow as tf
from keras import backend as K

INFEASIBLE_SCORE = -sys.float_info.max
LAMBDA=0

def tau_loss( tau ):
  def augmented_mse( score_data, D_pred ):
    # Determine which of Dpred correspond to fake val  
    neg_mask      = tf.equal(score_data,INFEASIBLE_SCORE)
    y_neg         = tf.boolean_mask(D_pred,neg_mask) 
    
    # Determine which of Dpred correspond to true fcn val
    pos_mask      = tf.not_equal(score_data,INFEASIBLE_SCORE)
    y_pos         = tf.boolean_mask(D_pred,pos_mask) 
    score_pos     = tf.boolean_mask(score_data,pos_mask)

    # compute mse w.r.t true function values
    mse_on_true_data = K.mean( (K.square(score_pos - y_pos)), axis=-1)
    return mse_on_true_data+tau[0]*K.mean( y_neg ) # try to minimize the value of y_neg
  return augmented_mse

def adv_mse( score_data, D_pred ):
  # Determine which of Dpred correspond to fake val  
  neg_mask      = tf.equal(score_data,INFEASIBLE_SCORE)
  y_neg         = tf.boolean_mask(D_pred,neg_mask) 
  
  # Determine which of Dpred correspond to true fcn val
  pos_mask      = tf.not_equal(score_data,INFEASIBLE_SCORE)
  y_pos         = tf.boolean_mask(D_pred,pos_mask) 
  score_pos     = tf.boolean_mask(score_data,pos_mask)

  # compute mse w.r.t true function values
  mse_on_true_data = K.mean( (K.square(score_pos - y_pos)), axis=-1)
  return mse_on_true_data+LAMBDA*K.mean( y_neg ) # try to minimize the value of y_neg

def unconstrained_mse( score_data, D_pred ):
  # Determine which of Dpred correspond to fake val  
  neg_mask      = tf.equal(score_data,INFEASIBLE_SCORE)
  y_neg         = tf.boolean_mask(D_pred,neg_mask) 
  
  # Determine which of Dpred correspond to true fcn val
  pos_mask      = tf.not_equal(score_data,INFEASIBLE_SCORE)
  y_pos         = tf.boolean_mask(D_pred,pos_mask) 
  score_pos     = tf.boolean_mask(score_data,pos_mask)

  # compute mse w.r.t true function values
  mse_on_true_data = K.mean( (K.square(score_pos - y_pos)), axis=-1)
  return mse_on_true_data+LAMBDA*(K.mean(y_neg) - K.mean(y_pos)) # try to minimize the value of y_neg

def hinge_mse( score_data,D_pred ):
  # Determine which of Dpred correspond to fake val  
  neg_mask      = tf.equal(score_data,INFEASIBLE_SCORE)
  y_neg         = tf.boolean_mask(D_pred,neg_mask) 
  
  # Determine which of Dpred correspond to true fcn val
  pos_mask      = tf.not_equal(score_data,INFEASIBLE_SCORE)
  y_pos         = tf.boolean_mask(D_pred,pos_mask) 
  score_pos     = tf.boolean_mask(score_data,pos_mask)

  # compute mse w.r.t true function values
  mse_on_true_data = K.mean( (K.square(score_pos - y_pos)), axis=-1)
  hinge_loss = K.max(y_neg - y_pos,0)
  return mse_on_true_data+LAMBDA*K.mean(hinge_mse) # try to minimize the value of y_neg



