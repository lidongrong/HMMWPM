# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:01:19 2023

@author: lidon
"""

import pandas as pd
import os
# import SD_generator as sdg
import matplotlib.pyplot as plt
import Adv_model as Model
import numpy as np
import multiprocessing as mp
import time
import os
import argparse
import torch
from AdvCirrUtils import *
import Adv_SD_generator as sdg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_path = 'D:/Object/PROJECTS/HMM/Cirrhosis Data/Granularity3/trans_aggregated'

covariates = np.array(['Intercept', 'AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG',
                       'LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
                       'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
                       'ARB', 'CalciumChannelBlocker', 'dys',
                       'dm', 'hypertension', 'Sex'])
covariate_types = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1,
                            1, 1, 1,
                            1, 1, 1])

sample_size = 4700
graph = np.array([[1, 1, 0, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]])

# use sdg to load data
data, lengths = sdg.data_loader(data_path, sample_size, covariates, covariate_types)
lengths = np.float32(lengths)
data = np.float32(data)
# drop empty sequences
l = lengths[lengths > 1]
data = data[lengths > 1]
lengths = np.int32(l)
# remove time point columns by converting it to intercept
m = Model.HMM_Model(data, lengths, covariates, covariate_types, graph=graph)
x = m.x
m.standardize()
m.add_intercept()
x, y = m.x, m.y

'''
# read data
print('reading data...')
sequences=data_reader(data_path,sample_size)
lengths=data_to_length(sequences)
sequences=data_to_tensor(sequences,stages)
# convert to numpy
lengths=lengths.numpy()
data=sequences.numpy()
# drop empty sequences
l=lengths[lengths>1]
data=data[lengths>1]
lengths=l
'''

if __name__ == '__main__':
    print('start working...')
    # m=Model.HMM_Model(data,lengths,covariates,covariate_types)
    # x,y=m.split()

    print('initializing parameters...')
    parser = argparse.ArgumentParser(description="tunning paramters")
    parser.add_argument("-batch", "--batch-size", default=40, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-hk", "--hk", default=1, type=float)
    parser.add_argument("-lbatch", "--latent-batch-size", default=1000, type=int)
    parser.add_argument("-SGLD", "--use-sgld", default=True, type=bool)
    parser.add_argument("-core", "--num-core", default=0, type=int)
    parser.add_argument("-beta", "--beta-prior", default='Normal', type=str)
    parser.add_argument("-alpha", "--alpha-prior", default='Normal', type=str)
    parser.add_argument("-normal-regularizer", "--normal-regularizer", default=1, type=float)
    parser.add_argument("-laplace-regularizer", "--laplace-regularizer", default=1, type=float)
    args = parser.parse_args(args=[])

    # adjust argparser
    args.batch_size = 12
    args.hk = 1
    args.latent_batch_size = 50
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1 / 126) * (1 / (0.5 * sample_size))
    args.use_sgld = False
    args.num_core = 1
    args.normal_regularizer = 1
    args.laplace_regularizer = 0.1
    args.beta_prior = 'Laplace'
    args.alpha_prior = 'Laplace'

    save_path = 'CirrDataAnalysis'

    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)
    for i in range(3, 3+8):
        print(f'running experiments {i}...')
        optimizer = Model.Random_Gibbs(model=m, args=args, initial=None)
        # param=optimizer.sys_scan(n=5000,log_step=25,prog_bar=True,prob=0.5,initial_x=None,initial_z=None)
        param = optimizer.random_scan(epoch=800, log_step=99999999, prog_bar=True, prob=0.5, initial_x=None,
                                      initial_z=None)
        # os.mkdir(f'NormalAnalysis{i}')
        os.mkdir(f'{save_path}/LaplaceAnalysis{i}')
        optimizer.pickle(f'{save_path}/LaplaceAnalysis{i}',None)
