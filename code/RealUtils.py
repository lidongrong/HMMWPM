# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:11:17 2023

@author: lidon
"""
import os
import pandas as pd
import numpy as np
import torch

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
    # drop stage column
    sequences[i]=sequences[i].drop(['Unnamed: 0','TimePoint'],axis=1)
    if 'Stage' in sequences[i].columns:
        sequences[i]=sequences[i].drop(['Stage'],axis=1)
    # standardize ages to avoid explosion
    if 'Age' in sequences[i].columns:
        sequences[i]['Age']=sequences[i]['Age']/100
    features=sequences[i].columns
    # standardize data
    for k in range(0,len(sequences[i].columns)-9):
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
stages=['HBeAg+ALT<=1ULN', 'HBeAg+ALT>1ULN', 'HBeAg-ALT<=1ULN',
                       'HBeAg-ALT>1ULN','Cirr','HCC','Death']