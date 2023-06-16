# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 12:55:38 2023

@author: lidon
"""

import os
import pandas as pd
import numpy as np
import torch


variates={'Lab': np.array(['AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG'],
       dtype=object),
 'Med': np.array(['LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
        'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
        'ARB', 'CalciumChannelBlocker'], dtype=object)}
        
covariates=np.array(['Intercept','AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG',
                     'LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
                            'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
                            'ARB', 'CalciumChannelBlocker','dys','dm','hypertension','Sex'])

def data_reader(data_path,sample_size):
  f=os.listdir(data_path)
  f=f[0:sample_size]
  for i in range(0,len(f)):
    f[i]=data_path+'/'+f[i]
  sequences=[]
  for k in range(0,sample_size):
    sequences.append(pd.read_csv(f[k]))
  return sequences

# convert data to tensor
# convert sequences to tensor
def data_to_tensor(sequences,stages):
  # sequences: must be a list of pd frames
  # stages: the observed stage, array of strings
  # stage in stages will be turned into
  # int types
  
  # first, acquire the maximum lengths
  lengths=data_to_length(sequences)
  max_length=np.int32(max(lengths))

  #features=sequences[0].columns
  for i in range(0,len(sequences)):
    # add intercept
    sequences[i].insert(0,'Intercept',np.ones(sequences[i].shape[0]))
    # drop stage column
    to_drop=['Stage','Unnamed: 0','TimePoint']
    for elem in to_drop:
      if elem in sequences[i].columns:
        sequences[i]=sequences[i].drop(elem,axis=1)
    # standardize ages to avoid explosion
    #sequences[i]['Age']=sequences[i]['Age']/100
    features=sequences[i].columns
    # standardize data
    for k in range(0,len(sequences[i].columns)-9):
      # only standardize non med features
      if not sequences[i].columns[k] in list(variates['Med'])+['Intercept']:
        seq=sequences[i][sequences[i].columns[k]]
        seq=(seq-seq.mean())/seq.std()
        sequences[i][sequences[i].columns[k]]=seq
    sequences[i]=sequences[i].values
    # if sequence[i] shorter than max_length
    # impute them with nan
    # but these imputed nan will be ignored
    # in training by specifications of lengths
    # of each sequence
    if sequences[i].shape[0]<max_length:
      compensate=np.empty((max_length-sequences[i].shape[0],len(features)))
      compensate[:]=np.nan
      sequences[i]=np.concatenate((sequences[i],compensate),axis=0)
    
    for k in range(0,len(sequences[i])):
      for r in range(0,len(stages)):
        if sequences[i][k][len(sequences[i][k])-1]==stages[r]:
          sequences[i][k][len(sequences[i][k])-1]=r
          break
  sequences=np.float32(sequences)
  sequences=torch.tensor(sequences)
  return sequences


def data_to_length(sequences):
  # sequences is the list of data frames
  # returned by data_reader
  # return a tensor, recording the length
  # of each array
  lengths=[]
  for k in range(0,len(sequences)):
    lengths.append(sequences[k].values.shape[0])
  return torch.tensor(lengths)
  
 
# all observable stages
stages=['Compensated','Decompensated','Recompensated','HCC','Death']