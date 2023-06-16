# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:00:03 2023

@author: lidon
"""

import matplotlib.pyplot as plt
import Adv_SD_generator as sdg
import Adv_model as Model
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\GraphSynthData\Rate0.3\PartialData'
param_path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\GraphSynthData'
hidden_data_path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\GraphSynthData\Rate0.5\HiddenStates'
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide'])
# graph regulation, prohibit illegal transition
graph=np.array([[1,1,0],[0,1,1],[1,0,1]])
#graph=np.ones((3,3))
# additional arguments
d=2
covariates=covariates[0:d]

types=np.ones(len(covariates))
sample_size=800
data,lengths=sdg.data_loader(path,sample_size,covariates,types)  
z=sdg.hidden_data_reader(hidden_data_path,sample_size)
beta=np.load('D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\GraphSynthData/beta.npy')


if __name__ == '__main__':
    
    m=Model.HMM_Model(data,lengths,covariates,types,graph=graph)
    x,y=m.split()
    
    parser=argparse.ArgumentParser(description="tunning paramters")
    parser.add_argument("-batch","--batch-size",default=40,type=int)
    parser.add_argument("-lr","--learning-rate",default=0.01,type=float)
    parser.add_argument("-hk","--hk",default=1,type=float)
    parser.add_argument("-lbatch","--latent-batch-size",default=1000,type=int)
    parser.add_argument("-SGLD","--use-sgld",default=True,type=bool)
    parser.add_argument("-core","--num-core",default=1,type=int)
    parser.add_argument("-beta","--beta-prior",default='Normal',type=str)
    parser.add_argument("-alpha","--alpha-prior",default='Normal',type=str)
    parser.add_argument("-normal-regularizer","--normal-regularizer",default=1,type=float)
    parser.add_argument("-laplace-regularizer","--laplace-regularizer",default=1,type=float)
    
    args=parser.parse_args(args=[])
    
    # adjust argparser
    args.batch_size=sample_size
    args.hk=1
    args.latent_batch_size=50
    args.use_sgld=False
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1/18)*(1/(0.2*sample_size))
    args.num_core=6
    args.beta_prior='Normal'
    args.alpha_prior='Normal'
    args.laplace_regularizer=0.1
    args.normal_regularizer=1
    
    optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
    
    
    true_param=optimizer.load_true_param(param_path)
    #optimizer.set_as_true_param(true_param)
    #optimizer.beta=beta
    
    
    mx,my,mz=optimizer.check_state(x,y,z)
    
    site=50
    zt=optimizer.sample_zt(x[site],y[site],z[site],optimizer.model.lengths[site],
                        optimizer.model.x_masks[site],optimizer.model.y_masks[site])
    
    #param=optimizer.run(n=40000,log_step=250,prog_bar=True,prob=1,initial_x=None,initial_z=z)
    #param=optimizer.random_scan(epoch=500,log_step=50,prog_bar=True,prob=0.5,initial_x=None,initial_z=None)
    param=optimizer.sys_scan(n=2000,log_step=25,prog_bar=True,prob=0.5,initial_x=None,initial_z=None)