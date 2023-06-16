# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:25:06 2023

@author: lidon
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
from utils import*
from RealUtils import*
import array_to_latex as a2l
import scipy.stats as ss
# posterior path
#path='D:/Files/CUHK_Material/Research_MakeThisObservable/EMR/Data/Cirrhosis Data/CirrPosterior/NormalAnalysis1'
path='D:/Files/CUHK_Material/Research_MakeThisObservable/EMR/Data/ETVTDF timeseries/LassoAnalysis1'

stages=['HBeAg+ALT<=1ULN',
'HBeAg+ALT>1ULN', 
'HBeAg-ALT<=1ULN',
'HBeAg-ALT>1ULN',
'Cirr',
'HCC',
'Death'
]

# stages for Cirrhosis Research
#stages=['Compensated','Decompensated','Recompensated','HCC','Death']

# covariates for ETVTDF Research
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide','Sex','Age'])

# covariates for Cirrhosis Research
'''
covariates=np.array(['Intercept','AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG',
                     'LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
                            'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
                            'ARB', 'CalciumChannelBlocker','dys',
                            'dm','hypertension','Sex'])
'''
# Load Data for ETVTDF research
'''
# align labels in ETVTDF Research
data_path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\ETVTDF timeseries\ETVTDF timeseries\DesignMatrix3'

covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide','Sex','Age'])
covariate_types=np.array(['numeric']*len(covariates))

sample_size=6000

# read data
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

m=Model.HMM_Model(data,lengths,covariates,covariate_types)
x,y=m.split()
'''
# Load Data for Cirrhosis research

data_path='D:/Files/CUHK_Material/Research_MakeThisObservable/EMR/Data/Cirrhosis Data/DesignMatrix/trans_aggregated'

covariates=np.array(['AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG',
                     'LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
                            'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
                            'ARB', 'CalciumChannelBlocker','dys',
                            'dm','hypertension','Sex'])
covariate_types=np.array(['numeric']*len(covariates))

sample_size=4700

# read data test
data,lengths=sdg.data_loader(data_path,sample_size,covariates,covariate_types)

# read data
'''
sequences=data_reader(data_path,sample_size)
lengths=data_to_length(sequences)
sequences=data_to_tensor(sequences,stages)
#convert to numpy
lengths=lengths.numpy()
data=sequences.numpy()
'''
lengths=np.float32(lengths)
data=np.float32(data)
# drop empty sequences
l=lengths[lengths!=0]
data=data[lengths!=0]
lengths=l
# drop time point column
data=np.delete(data,0,2)
m=Model.HMM_Model(data,lengths,covariates,covariate_types)
x,y=m.split()

# data standardizer
tmp=x.reshape(x.shape[0]*x.shape[1],x.shape[2])
upperBound=np.nanmax(tmp,axis=0)
lowerBound=np.nanmin(tmp,axis=0)
tmp=(tmp-lowerBound)/upperBound
tmp=tmp.reshape((x.shape[0],x.shape[1],x.shape[2]))
x=np.float32(tmp)

# pairwise plots
features=[]
for k in range(len(covariates)):
    f=x[:,:,k]
    f=np.concatenate(f,axis=0)
    features.append(f)
features=np.array(features)
col=np.nanmean(features,axis=1)
# add little disturbance to ensure corr exists
features=features+np.random.normal(0,0.01,features.shape)

corr=np.zeros((len(covariates),len(covariates)))
pValue=np.zeros((len(covariates),len(covariates)))
for i in range(len(covariates)):
    for j in range(len(covariates)):
        axis=~np.isnan(features[i])
        axis = axis * ~np.isnan(features[j])
        print(i,j,np.sum(axis))
        psr=ss.pearsonr(features[i][axis],features[j][axis])
        #print(i,j,psr[0] )
        corr[i][j]=psr[0]
        pValue[i][j]=psr[1]
sns.heatmap(corr,xticklabels=covariates,yticklabels=covariates)
# filter significant entries
sig=np.zeros(corr.shape)
sig[corr>0.7]=corr[corr>0.7]
sns.heatmap(cov,xticklabels=covariates,yticklabels=covariates)

xs={}
for k in range(len(covariates)):
    xs[covariates[k]]=features[k]
xs=pd.DataFrame(xs)
xs=xs.fillna(method='ffill')
xs=xs.fillna(method='bfill')
designMatrix=xs.to_numpy()
gram= np.linalg.inv(designMatrix.T @ designMatrix)
# calculate VIF
VIF=np.zeros((len(covariates)))
for i in range(len(covariates)):
    VIF[i]=gram[i][i] * (len(features[0]))*np.std(designMatrix[i])*np.std(designMatrix[i])

VIFMap={}
for k in range(len(covariates)):
    VIFMap[covariates[k]]=np.array([VIF[k]])

# filter the colinear parts
VIFMap=pd.DataFrame(VIFMap)
colLinearPart=VIFMap.iloc[0][VIFMap.iloc[0]>2]
colLinearIndex=list(colLinearPart.index)
colLinearX=xs[colLinearIndex]
