# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:31:11 2023

@author: s1155151972
"""


import pandas as pd
import os
import SD_generator as sdg
import matplotlib.pyplot as plt
import Adv_model as Model
import numpy as np
import multiprocessing as mp
import time
import os
import argparse
import torch
from Adv_RealUtils import*
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_path='trans_aggregated'

covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide','Sex','Age'])
covariate_types=np.array(['numeric']*len(covariates))

sample_size=12500

# read data
print('reading data...')
sequences=data_reader(data_path,sample_size)
lengths=data_to_length(sequences)
sequences=data_to_tensor(sequences,stages)
# convert to numpy
lengths=lengths.numpy()
data=sequences.numpy()
# drop empty sequences
l=lengths[lengths!=0]
data=data[lengths!=0]
lengths=l

if __name__ == '__main__':
    print('start working...')
    m=Model.HMM_Model(data,lengths,covariates,covariate_types)
    x,y=m.split()
    
    print('initializing parameters...')
    parser=argparse.ArgumentParser(description="tunning paramters")
    parser.add_argument("-batch","--batch-size",default=40,type=int)
    parser.add_argument("-lr","--learning-rate",default=0.01,type=float)
    parser.add_argument("-hk","--hk",default=1,type=float)
    parser.add_argument("-lbatch","--latent-batch-size",default=1000,type=int)
    parser.add_argument("-SGLD","--use-sgld",default=True,type=bool)
    parser.add_argument("-core","--num-core",default=0,type=int)
    args=parser.parse_args(args=[])
    
    # adjust argparser
    args.batch_size=12
    args.hk=1
    args.latent_batch_size=1500
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1/126)*(1/(0.25*sample_size))
    args.use_sgld=False
    args.num_core=96
    
    
    for i in range(1,8):
        print(f'running experiments {i}...')
        optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
        param=optimizer.sys_scan(n=5000,log_step=9999999,prog_bar=True,prob=0.5,initial_x=None,initial_z=None)
        os.mkdir(f'RealSimulation{i}')
        optimizer.pickle(f'RealSimulation{i}',None)

