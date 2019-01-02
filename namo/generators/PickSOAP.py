import pickle
import math
import csv
import shutil
import os
import random 
import pandas as pd
from generators.Uniform import UniformPlace
from NAMO_env import NAMO
from data_load_utils import get_sars_data
from openravepy import *
from sklearn.preprocessing import StandardScaler
from generators.PickSOAP import *

from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.objectives import *
from keras import backend as K
from keras import initializers
from Q_loss_functions import *
from generators.Policy import Policy,noise,G_loss
from slice_functions import *

# misc vector slices
# misc_vector = [c0,opose,oshape]
# misc = np.r_[c0,opose,oshape]


class PickSOAP(Policy):
  def __init__(self,\
               sess,
               dim_a,\
               dim_misc,\
               dim_cvec,\
               weight_dir,\
               key_configs,\
               Qloss,
               d_lr,\
               g_lr,\
               n_score,\
               explr_const,\
               tau,architecture):
    super(PickSOAP,self).__init__(sess,\
                                  dim_a,\
                                  dim_misc,\
                                  dim_cvec,\
                                  weight_dir,\
                                  key_configs,\
                                  Qloss,\
                                  d_lr,\
                                  g_lr,\
                                  n_score,
                                  explr_const,\
                                  tau,architecture)
  def createDisc(self):
    init_ = self.initializer
    dense_num = 64
    n_filters = 64

    self.dim_misc = 6

    shared_l1 = Dense(dense_num,activation='relu')
    R_xy = Lambda(slice_rxy)(self.a_input)
    O_xy = Lambda(slice_oxy)(self.misc_input)
    global_xy = Add()([R_xy,O_xy])
    R_th = Lambda(slice_rth)(self.a_input)
    Wpose_H = Concatenate(axis=1)([global_xy,R_th]) # pose feature
    Wpose_H = shared_l1(Wpose_H)

    W_c0    = Lambda(slice_c0)(self.misc_input)
    W_c0_H  = shared_l1(W_c0)
  
    W_H = Concatenate(axis=1)([W_c0_H,Wpose_H])
    W_H = RepeatVector(self.n_key_confs)(W_H)
    W_H = Reshape((self.n_key_confs,dense_num*2,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+dense_num*2),\
               strides=(1,1),
               activation='relu')(WC_H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)

    # Grasp 
    R_grasp = Lambda(slice_grasp)(self.a_input)
    O_shape = Lambda(slice_shape)(self.misc_input)
    Wgrasp  = Concatenate(axis=1)([R_grasp,O_shape]) # pose feature

    # Combined pose and grasp
    H = Concatenate(axis=1)([H,Wgrasp])
    H = Dense(dense_num,activation='relu')(H)

    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input,self.tau_input],\
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
    return disc

  def createGen(self):
    init_       = self.initializer
    dense_num   = 64
    n_filters   = 64
    n_key_confs = self.n_key_confs
    self.dim_misc = 6

    O_xy = Lambda(slice_oxy)(self.misc_input)
    Wpose_H = Dense(dense_num,activation='relu')(O_xy)

    W_c0    = Lambda(slice_c0)(self.misc_input)
    W_c0_H  = Dense(dense_num,activation='relu')(W_c0)
  
    W_H = Concatenate(axis=1)([W_c0_H,Wpose_H])
    W_H = RepeatVector(self.n_key_confs)(W_H)
    W_H = Reshape((self.n_key_confs,dense_num*2,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+dense_num*2),\
               strides=(1,1),
               activation='relu')(WC_H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc1(self):
    init_ = self.initializer
    dense_num = 64
    n_filters = 64

    self.dim_misc = 6

    shared_l1 = Dense(dense_num,activation='relu')
    R_xy = Lambda(slice_rxy)(self.a_input)
    O_xy = Lambda(slice_oxy)(self.misc_input)
    global_xy = Add()([R_xy,O_xy])
    R_th = Lambda(slice_rth)(self.a_input)
    Wpose_H = Concatenate(axis=1)([global_xy,R_th]) # pose feature
    Wpose_H = shared_l1(Wpose_H)

    W_c0    = Lambda(slice_c0)(self.misc_input)
    W_c0_H  = shared_l1(W_c0)
  
    W_H = Concatenate(axis=1)([W_c0_H,Wpose_H])
    W_H = RepeatVector(self.n_key_confs)(W_H)
    W_H = Reshape((self.n_key_confs,dense_num*2,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+dense_num*2),\
               strides=(1,1),
               activation='relu')(WC_H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)

    # Grasp 
    R_grasp = Lambda(slice_grasp)(self.a_input)
    O_shape = Lambda(slice_shape)(self.misc_input)
    Wgrasp  = Concatenate(axis=1)([R_grasp,O_shape]) # pose feature

    # Combined pose and grasp
    H = Concatenate(axis=1)([H,Wgrasp])
    H = Dense(dense_num,activation='relu')(H)

    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input,self.tau_input],\
                  output=[disc_output],\
                  name='disc_output')
    disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
    return disc

  def createGen1(self):
    init_       = self.initializer
    dense_num   = 64
    n_filters   = 64
    n_key_confs = self.n_key_confs
    self.dim_misc = 6

    O_xy = Lambda(slice_oxy)(self.misc_input)
    Wpose_H = Dense(dense_num,activation='relu')(O_xy)

    W_c0    = Lambda(slice_c0)(self.misc_input)
    W_c0_H  = Dense(dense_num,activation='relu')(W_c0)
  
    W_H = Concatenate(axis=1)([W_c0_H,Wpose_H])
    W_H = RepeatVector(self.n_key_confs)(W_H)
    W_H = Reshape((self.n_key_confs,dense_num*2,1))(W_H)

    C_H = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,4+dense_num*2),\
               strides=(1,1),
               activation='relu')(WC_H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1  = Conv2D(filters=n_filters,
                kernel_size=(1,1),
                strides=(1,1),
                activation='relu')(H)
    H1=H
    H = MaxPooling2D(pool_size=(2,1))(H1)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    Z_H = Dense(dense_num,activation='relu')(self.z)
    H = Concatenate()([H,Z_H])
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output





