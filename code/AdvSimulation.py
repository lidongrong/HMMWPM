# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 16:03:11 2023

@author: lidon
"""


import Adv_SD_generator as sdg
import Adv_model as Model
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import argparse
import os
import utils

rate=[0,0.3,0.5,0.6,0.7,0.8,0.9]

data_path='D:/Object/PROJECTS/HMM/AdvancedModelCode/SparseSynthData'
path=[f'{data_path}/Rate0.5/FullData',
      f'{data_path}/Rate0.3/PartialData',
      f'{data_path}/Rate0.5/PartialData',
      f'{data_path}/Rate0.6/PartialData',
      f'{data_path}/Rate0.7/PartialData',
      f'{data_path}/Rate0.8/PartialData',
      f'{data_path}/Rate0.9/PartialData']
param_path=data_path
real_path=f'{data_path}/Rate0.5/FullData'
parent='D:/Object/PROJECTS/HMM/AdvancedModelCode/SparseLaplaceResult800'
save_path=[f'rate{rate[i]}' for i in range(len(rate))]
hidden_data_path=f'{data_path}/Rate0.5/HiddenStates'



covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide'])
graph=np.ones((3,3))
types=np.array(['numeric','dummy'])
d=8
covariates=covariates[0:d]
types=np.ones(len(covariates))



sample_size=800
num=16
iteration=7500
prob=0.6
log_step=5000000
latent_batch_size=80
num_core=6

z=sdg.hidden_data_reader(hidden_data_path,sample_size)
real_data,real_lengths=sdg.data_loader(real_path,sample_size,covariates,types)
#data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)


if __name__ == '__main__':
    # define arg parser
    parser = argparse.ArgumentParser(description="tunning paramters")
    parser.add_argument("-batch", "--batch-size", default=40, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-hk", "--hk", default=1, type=float)
    parser.add_argument("-lbatch", "--latent-batch-size", default=1000, type=int)
    parser.add_argument("-SGLD", "--use-sgld", default=True, type=bool)
    parser.add_argument("-core", "--num-core", default=1, type=int)
    parser.add_argument("-beta", "--beta-prior", default='Normal', type=str)
    parser.add_argument("-alpha", "--alpha-prior", default='Normal', type=str)
    parser.add_argument("-normal-regularizer", "--normal-regularizer", default=1, type=float)
    parser.add_argument("-laplace-regularizer", "--laplace-regularizer", default=1, type=float)

    args = parser.parse_args(args=[])

    # adjust argparser
    args.batch_size = sample_size
    args.hk = 1
    args.latent_batch_size = 50
    args.use_sgld = False
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1 / 18) * (1 / (0.2 * sample_size))
    args.num_core = 4
    args.beta_prior = 'Laplace'
    args.alpha_prior = 'Laplace'
    args.laplace_regularizer = 0.1
    args.normal_regularizer = 1
    
    #opts=[[] for i in range(len(path))]
    if os.path.exists(parent):
        pass
    else:
        utils.make_category(parent,save_path)

    for j in range(num):
        for i in range(len(path)):
            print(f'Running Model with sample size = f{sample_size} and regularizer = {args.alpha_prior}...')
            print('from path: ', path[i])
            print('Will save to path: ', f'{parent}/{save_path[i]}/Simulation{j}')
            print('loading data...')
            data,lengths=sdg.data_loader(path[i],sample_size,covariates,types)
            z=sdg.hidden_data_reader(hidden_data_path,sample_size)
            print('building model...')
            m=Model.HMM_Model(data,lengths,covariates,types,graph=graph)
            # load real data
            print('loading real data...')
            real_m=Model.HMM_Model(real_data,real_lengths,covariates,types,graph=graph)
            real_x,real_y=real_m.split()
            real_z=z

            print('preparing running...')
            optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
            optimizer.set_real(real_x,z)

            print('sampling...')
            param=optimizer.sys_scan(n=iteration,log_step=log_step,prog_bar=True,prob=prob,initial_x=None,initial_z=None)
            #opts[i].append(optimizer)
            os.mkdir(f'{parent}/{save_path[i]}/Simulation{j}')
            print('start pickling...')
            optimizer.pickle(f'{parent}/{save_path[i]}/Simulation{j}',param_path)
            
            del optimizer
            del param
    
  