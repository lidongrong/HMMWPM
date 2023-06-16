# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 23:17:46 2023

@author: lidon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:07:48 2023

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
# effective numbers
e=4
# effective number for var
v=10
# stage for ETVTDF Research

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

pi=np.load(f'{path}/pi.npy')
mu=np.load(f'{path}/mu.npy')
beta=np.load(f'{path}/beta.npy')
sigma=np.load(f'{path}/sigma.npy')
alpha=np.load(f'{path}/alpha.npy')

mean=pd.read_csv(f'{path}/posterior_mean.csv')
var=pd.read_csv(f'{path}/posterior_var.csv')

# discard burn in period
post_pi=pi[6000:]
post_mu=mu[6000:]
post_beta=beta[6000:]
post_sigma=sigma[6000:]
post_alpha=alpha[6000:]

# calculate post mean 
mean_pi=np.round(sum(post_pi)/len(post_pi),e)
mean_mu=np.round(sum(post_mu)/len(post_mu),e)
mean_beta=np.round(sum(post_beta)/len(post_beta),e)
mean_sigma=np.round(sum(post_sigma)/len(post_sigma),e)
mean_alpha=np.round(sum(post_alpha)/len(post_alpha),e)

# Switch the labels
# The way to switch labels: count empirical starting values, and permute estimation to match
# the empirical starting values

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

# count starting distribution
tmp=np.array([sum(y[:,0,i]==1) for i in range(7)])
tmp=tmp/sum(tmp)

perm=find_permute(mean_pi,tmp)
#perm=[[5, 6, 0, 3, 4, 1, 2]]
post_pi,post_alpha,post_mu,post_sigma,post_beta=permute_train(perm,post_pi,post_alpha,post_mu,post_sigma,post_beta)

#stages=np.array(stages)
#stages=stages[perm]


# Align Labels in Cirrhosis Research
'''
data_path='D:/Files/CUHK_Material/Research_MakeThisObservable/EMR/Data/Cirrhosis Data/DesignMatrix/trans_aggregated'

covariates=np.array(['AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG',
                     'LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
                            'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
                            'ARB', 'CalciumChannelBlocker','dys',
                            'dm','hypertension','Sex'])
covariate_types=np.array(['numeric']*len(covariates))

sample_size=4700

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

# count starting distribution
tmp=np.array([sum(y[:,0,i]==1) for i in range(5)])
tmp=tmp/sum(tmp)

perm=find_permute(mean_pi,tmp)
#perm=[0, 1, 4, 2, 3]
post_pi,post_alpha,post_mu,post_sigma,post_beta=permute_train(perm,post_pi,post_alpha,post_mu,post_sigma,post_beta)
covariates=np.array(['Intercept','AFP', 'Cr', 'PLT', 'HBeAg', 'GGT', 'HDL', 'LDL', 'TC', 'TG',
                     'LipidLoweringAgent', 'BetaBlocker', 'IS', 'Cytotoxic', 'ACEI',
                            'Insulin', 'Antiplatelet', 'OHA', 'Thiazide', 'Anticoagulant',
                            'ARB', 'CalciumChannelBlocker','dys',
                            'dm','hypertension','Sex'])
#stages=np.array(stages)
#stages=stages[perm]
'''

#perm=[3, 0, 5, 4, 2, 1, 6]
#post_pi,post_transition,post_mu,post_sigma,post_beta=permute_train(perm,post_pi,post_transition,post_mu,post_sigma,post_beta)

# calculate post mean 
mean_pi=np.round(sum(post_pi)/len(post_pi),e)
mean_mu=np.round(sum(post_mu)/len(post_mu),e)
mean_beta=np.round(sum(post_beta)/len(post_beta),e)
mean_sigma=np.round(sum(post_sigma)/len(post_sigma),e)
mean_alpha=np.round(sum(post_alpha)/len(post_alpha),e)

# compute post variance
var_pi=np.round(np.var(post_pi,axis=0),v)
var_mu=np.round(np.var(post_mu,axis=0),v)
var_beta=np.round(np.var(post_beta,axis=0),v)
var_sigma=np.round(np.var(post_sigma,axis=0),v)
var_alpha=np.round(np.var(post_alpha,axis=0),v)


# to this step, we've finished all data preprocessing, start analyzing


# plot
#sns.heatmap(mean_transition)

# plot transition
dick={#'pi':np.expand_dims(post_pi,0),
      f'transition{i}': np.expand_dims(post_transition[:,i,:],0) for i in range(7)
      }
dick['pi']=np.expand_dims(post_pi,0)
dataset=az.convert_to_inference_data(dick)

fig=az.plot_posterior(dataset,var_names='pi')
#fig = fig.flatten()[0].get_figure()
#fig.show()
for i in range(7):
    az.plot_posterior(dataset,var_names=f'transition{i}')
#az.plot_posterior(dataset,var_names='lower_transition')


# process mu
mu_map={}
for i in range(mean_mu.shape[0]):
    mu_map[f'{stages[i]}']={}
    for j in range(mean_mu.shape[1]):
        #mu_map[f'{stages[i]},{covariates[j]}']=[mean_mu[i,j],var_mu[i,j]]
        mu_map[f'{stages[i]}'][f'{covariates[j]}']=[mean_mu[i,j],var_mu[i,j]]
        
for i in range(mean_mu.shape[0]):
    tmp_map=pd.DataFrame(mu_map[f'{stages[i]}'])
    tmp_map=tmp_map.T
    tmp_map.to_csv(f'mu_stages{i}.csv')


beta_map={}
for i in range(mean_beta.shape[0]):
    for j in range(mean_beta.shape[1]):
        beta_map[f'stage{i}->stage{j}']={}
        for k in range(mean_beta.shape[2]):
            beta_map[f'stage{i}->stage{j}'][f'{covariates[k]}']=[mean_beta[i,j,k],var_beta[i,j,k]]
for i in range(mean_beta.shape[0]):
    for j in range(mean_beta.shape[1]):
        tmp_map=pd.DataFrame(beta_map[f'stage{i}->stage{j}'])
        tmp_map=tmp_map.T
        tmp_map.to_csv(f'Stage{i}ToStage{j}.csv')

beta_map=pd.DataFrame(beta_map)
beta_map=beta_map.T
beta_map.columns=['Estimated Posterior Mean','Estimated Posterior Var']


covariates=np.array(['Intercept','AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide','dys','dm','hypertension','Sex','Age'])

# find top 5 infectors
def topN(a,n):
    # function that finds top n values in array a
    # return indices corresponds to values from large to small
    ind=np.argpartition(np.abs(a),-n)[-n:]
    ind[np.argsort(a[ind])]
    return ind

# align with respect to the going back to self-transition
a=mean_alpha.copy()
for i in range(a.shape[0]):
    a[i]=a[i]-a[i][i]
mean_alpha=a

# find top k factors and corresponding variance
k=5
transition={}
for i in range(mean_alpha.shape[0]):
    for j in range(mean_alpha.shape[1]):
        ind=topN(mean_alpha[i][j],k)
        transition[f'{stages[i]}->{stages[j]}']=pd.DataFrame({f'{stages[i]}->{stages[j]}':covariates[ind],
                                                 'feature':mean_alpha[i][j][ind],
                                                 'variance':var_alpha[i][j][ind],
                                                 'z score': 2*(1-ss.norm.cdf(
                                                     np.abs(mean_alpha[i][j][ind])/np.sqrt(var_alpha[i][j][ind])))})



TransTable={k:[] for k in covariates}
TransTable['Transition']=[]
for i in range(mean_alpha.shape[0]):
    for j in range(mean_alpha.shape[1]):
        TransTable['Transition'].append(f'{stages[i]}->{stages[j]}')
        for k in range(len(covariates)):
            TransTable[covariates[k]].append(mean_alpha[i][j][k]) 
TransTable=pd.DataFrame(TransTable)


# pairwise plots
features=[]
for k in range(len(covariates)):
    f=x[:,:,k]
    f=np.concatenate(f,axis=0)
    features.append(f)
features=np.array(features)
col=np.nanmean(features,axis=1)

corr=np.zeros((len(covariates),len(covariates)))
pValue=np.zeros((len(covariates),len(covariates)))
for i in range(len(covariates)):
    for j in range(len(covariates)):
        axis=~np.isnan(features[i])
        axis = axis * ~np.isnan(features[j])
        psr=ss.pearsonr(features[i][axis],features[j][axis])
        corr[i][j]=psr[0]
        pValue[i][j]=psr[1]
        

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
