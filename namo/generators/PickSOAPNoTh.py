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
from Policy import Policy

class PickSOAPNoTh(Policy):
  def __init__(self,sess,
                dim_a,\
                dim_misc,\
                dim_cvec,\
                weight_dir,\
                key_configs,\
                Qloss,
                d_lr,\
                g_lr,n_score):
  
    super(PickSOAPNoTh,self).__init__(sess,\
                                   dim_a,\
                                   dim_misc,\
                                   dim_cvec,\
                                   weight_dir,\
                                   key_configs,\
                                   Qloss,\
                                   d_lr,\
                                   g_lr,n_score)

  def createGen(self):
    init_       = self.initializer
    dense_num   = 64
    n_filters   = 64
    n_key_confs = self.n_key_confs

    W_H = Dense(dense_num,activation='relu')(self.misc_input)
    W_H = Dense(dense_num,activation='relu')(W_H)
    W_H = RepeatVector(self.n_key_confs)(W_H)
    W_H = Reshape( (self.n_key_confs,dense_num,1))(W_H)
    C_H  = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,dense_num+self.dim_cvec[-1]),\
               strides=(1,1),
               activation='relu')(WC_H)

    """
    W    = RepeatVector(self.n_key_confs)(self.misc_input)
    W_H  = Reshape( (self.n_key_confs,self.dim_misc,1))(W)
    C_H  = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    WC_H = Concatenate(axis=2)([W_H,C_H])
    H = Conv2D(filters=n_filters,\
               kernel_size=(1,self.dim_cvec[-1]+self.dim_misc),\
               strides=(1,1),
               activation='relu')(WC_H)
    """
    H = Conv2D(filters=n_filters,
               kernel_size=(1,1),
               strides=(1,1),
               activation='relu')(H)
    H = Conv2D(filters=n_filters,
               kernel_size=(1,1),
               strides=(1,1),
               activation='relu')(H)
    H = Conv2D(filters=n_filters,
               kernel_size=(1,1),
               strides=(1,1),
               activation='relu')(H)
    H = Conv2D(filters=n_filters,
               kernel_size=(1,1),
               strides=(1,1),
               activation='relu')(H)
    H = MaxPooling2D(pool_size=(2,1))(H)
    H = Flatten()(H)
    H = Dense(dense_num,activation='relu')(H)
    H = Dense(dense_num,activation='relu')(H)
    Z_H = Dense(dense_num,activation='relu')(self.z)
    H   = Concatenate()([H,Z_H])
    a_gen_output = Dense(self.dim_a,
                         activation='linear',
                         init=init_,
                         name='a_gen_output')(H) 
    a_gen = Model(input=[self.z,self.misc_input,self.c_input], output=a_gen_output)
    return a_gen,a_gen_output

  def createDisc(self):
    init_ = self.initializer
    dense_num = 64
    n_filters = 64
    XW   = Concatenate(axis=1)([self.a_input,self.misc_input])
    XW_H = Dense(dense_num,activation='relu')(XW)
    XW_H = Dense(dense_num,activation='relu')(XW_H)
    XW_H = RepeatVector(self.n_key_confs)(XW_H)
    XW_H = Reshape( (self.n_key_confs,dense_num,1))(XW_H)
    C_H  = Reshape( (self.n_key_confs,self.dim_cvec[-1],1))(self.c_input)
    XC_H = Concatenate(axis=2)([XW_H,C_H])

    H = Conv2D(filters=n_filters,\
               kernel_size=(1,dense_num+self.dim_cvec[-1]),\
               strides=(1,1),
               activation='relu')(XC_H)
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
    disc_output = Dense(1, activation='linear',init=init_)(H)
    disc = Model(input=[self.a_input,self.misc_input,self.c_input],\
                  output=[disc_output],\
                  name='disc_output')
    if self.Qloss == 'adv':
      disc.compile(loss=adv_mse, optimizer=self.opt_D)
    elif self.Qloss == 'unconstrained':
      disc.compile(loss=unconstrained_mse, optimizer=self.opt_D)
    elif self.Qloss == 'hinge':
      disc.compile(loss=hinge_mse, optimizer=self.opt_D)
    return disc


