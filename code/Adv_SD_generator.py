# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:59:25 2023

@author: lidon
"""

import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import os
from sklearn.preprocessing import normalize

# patient record
class Record:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z


class Synthesize:
    """
    generate synthetic data
    """
    def __init__(self,covariates=None,pi=None,beta=None,mu=None,sigma=None,alpha=None,num=None,lbd=100,graph=None):
        '''
        covariates: demographic covariates of patients
        pi: initial distribution
        transition: transition matrix
        beta: regression coefficients
        mu: mean of conditional distribution of y|z
        sigma: covariance matrix of conditional distribution of y|z. i.e. y|z ~ N(mu_z,sigma_z)
        num: patient number
        lbd: poisson distribution parameter, used to control average length of each seq
        graph: transitional graph, with graph[i][j]=0 meaning transition from i to j prohibited
        '''
        self.covariates=covariates
        self.pi=pi
        #self.transition=transition
        self.beta=beta
        self.mu=mu
        self.sigma=sigma
        self.alpha=alpha
        self.num=num
        self.lbd=lbd
        # record and censored record
        self.emr=None
        self.partial_emr=None
        if graph is None:
            self.graph=np.ones((len(self.pi),len(self.pi)))
        else:
            self.graph=graph
        # prohibit impossible transition
        
        for i in range(len(self.pi)):
            for j in range(len(self.pi)):
                if self.graph[i][j]==0:
                    self.alpha[i][j][:]=0
        
    
    def generate_sequences(self):
        '''
        generate fully observed sequences
        '''
        full_patient=[]
        latent_size=len(self.pi)
        # generate sequences one by one
        
        # hidden state
        for i in range(self.num):
            length=np.random.poisson(self.lbd,1)[0]
            length=max(10,length)
            assert length>0
            # hidden state
            z=np.zeros((length,latent_size))
            x=np.zeros((length,self.mu.shape[1]))
            y=np.zeros((length,latent_size))
            for j in range(length):
                # first generate hidden states
                if j==0:
                    state=np.random.choice(latent_size,1,True,p=self.pi)[0]
                    new_z=[0]*latent_size
                    new_z[state]=1
                    z[j]=new_z
                else:
                    state=np.random.choice(latent_size,1,True,p=transition[state]/sum(transition[state]))[0]
                    new_z=[0]*latent_size
                    new_z[state]=1
                    z[j]=new_z
                    
                # Then generate x from z:
                # acquire corresponding mu and sigma
                tmp_mu=self.mu[state]
                tmp_sigma=self.sigma[state]
                new_x=np.random.multivariate_normal(tmp_mu,tmp_sigma,1)[0]
                x[j]=new_x
                
                # finally generate y from x and z
                # beta_z
                tmp_beta=self.beta[state]
                # logistic regression probability
                prob=np.exp(np.dot(tmp_beta,new_x))
                prob=np.append(prob,1)
                prob=prob/sum(prob)
                #print(prob)
                y_state=np.random.choice(latent_size,1,True,p=prob)[0]
                new_y=[0]*latent_size
                new_y[y_state]=1
                y[j]=new_y
                
                # now construct the transition matrix
                # at first, alpha has shape (7,6,40)
                # insert a line with all zeros to the second dimension to make it 
                # have shape (7,7,40)
                a=self.alpha
                # now a.shape=(7,7,40)
                log_transition=a @ new_x
                transition=np.exp(log_transition)*self.graph
                
                
            x=np.array(x)
            #x=x.swapaxes(1,0)
            #x=x.T
            y=np.array(y)
            z=np.array(z)
            y=y.astype(np.float32)
            z=z.astype(np.float32)
            new_record=Record(x,y,z)
            full_patient.append(new_record)
        self.emr=full_patient
        return full_patient
    
    # p is the missing rate
    def generate_partial_sequences(self,p):
        '''
        generate partially observed sequences
        with missing rate p
        observations will be set as np.nan with prob p
        '''
        assert self.emr
        full_patient=self.emr
        partial_obs=[]
        for i in range(self.num):
            pat=full_patient[i]
            padding=np.random.choice([np.nan,1],pat.x.shape,True,[p,1-p])
            partial_x=pat.x*padding
            partial_y=pat.y.copy()
            for j in range(partial_y.shape[0]):
                if np.random.choice([1,0],1,True,[p,1-p])[0]:
                    partial_y[j]=[np.nan]*len(self.pi)
            partial_obs.append(Record(partial_x,partial_y,pat.z))
        self.partial_emr=partial_obs
        return partial_obs
    
    def save_data(self,path=''):
        '''
        save data to path specified
        3 folders will be constructed: the first folder stores true states z
        The second folder stores full data 
        The third folder stores partial data
        '''
        os.mkdir(f'{path}/HiddenStates')
        os.mkdir(f'{path}/FullData')
        os.mkdir(f'{path}/PartialData')
        hidden_path=f'{path}/HiddenStates'
        full_path=f'{path}/FullData'
        partial_path=f'{path}/PartialData'
        # make sure exists
        assert self.partial_emr and self.emr
        for i in range(self.num):
            # first store hidden data
            hidden_data=self.emr[i].z
            hidden_data=pd.DataFrame(hidden_data)
            hidden_data.to_csv(f'{hidden_path}/{i}.csv')
            
            # then store full data
            x=self.emr[i].x
            y=self.emr[i].y
            d={self.covariates[i]:x[:,i] for i in range(len(self.covariates))}
            for j in range(len(self.pi)):
                d[j]=y[:,j]
            
            full_d=pd.DataFrame(d)
            full_d.to_csv(f'{full_path}/{i}.csv')
            
            # finally store partial data
            x=self.partial_emr[i].x
            y=self.partial_emr[i].y
            d={self.covariates[i]:x[:,i] for i in range(len(self.covariates))}
            for j in range(len(self.pi)):
                d[j]=y[:,j]
            
            partial_d=pd.DataFrame(d)
            partial_d.to_csv(f'{partial_path}/{i}.csv')
    

# convert data to matrix
def data_to_array(sequences,covariates,covariate_types,standardize=False):
    '''
    convert sequences to nd arrays
    sequences must be raw pandas frames read from pre-specified paths
    covariates is a list of strings indicating the name of covariates
    covariate_types: indicate if each covariate is numeric or dummy
    standardize: if standardize data
    '''
    lengths=data_to_length(sequences)
    max_length=np.int32(max(lengths))
    
    for i in range(0,len(sequences)):
        features=sequences[i].columns
        # standardize data
        sequences[i]=sequences[i].drop(['Unnamed: 0'],axis=1)
        if standardize:
            for k in range(0,len(covariates)):
                if covariate_types[k]=='numeric':
                    seq=sequences[i][sequences[i].columns[k]]
                    seq=(seq-min(seq))/(max(seq)-min(seq))
                    sequences[i][sequences[i].columns[k]]=seq
        
        sequences[i]=sequences[i].values
        if sequences[i].shape[0]<max_length:
            features=sequences[i].shape[1]
            compensate=np.empty((max_length-sequences[i].shape[0],features))
            compensate[:]=np.nan
            sequences[i]=np.concatenate((sequences[i],compensate),axis=0)
    
    #sequences=np.float32(sequences)
    sequences=np.array(sequences)
    return sequences

def data_reader(data_path,sample_size):
    '''
    read data from data_path with sample_size
    '''
    f=os.listdir(data_path)
    f=f[0:sample_size]
    for i in range(0,len(f)):
        f[i]=data_path+'/'+f[i]
    sequences=[]
    for k in range(0,sample_size):
        sequences.append(pd.read_csv(f[k]))
    return sequences

# convert data to their respective length
def data_to_length(sequences):
    '''
    sequences should be raw pandas frames read from the pre-specified paths
    '''
    lengths=[]
    for k in range(len(sequences)):
        lengths.append(sequences[k].values.shape[0])
    return np.array(lengths)

# load data
def data_loader(data_path,sample_size,covariates,covariate_types):
    '''
    load data into environment
    data_path: path of data
    sample_size: how many sample to load
    covariates: list of string of covariate names
    '''
    sequences=data_reader(data_path,sample_size)
    lengths=data_to_length(sequences)
    sequences=data_to_array(sequences,covariates,covariate_types)
    return sequences,lengths

def hidden_data_reader(data_path,sample_size):
    '''
    test code:
    hidden_data_path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\SynData\Rate0.5\HiddenStates'
    sample_size=50
    z=hidden_data_reader(hidden_data_path,sample_size)
    '''
    f=os.listdir(data_path)
    f=f[0:sample_size]
    for i in range(0,len(f)):
        f[i]=data_path+'/'+f[i]
    sequences=[]
    for k in range(0,sample_size):
        sequences.append(pd.read_csv(f[k]))
    
    x=data_reader(data_path,sample_size)
    lengths=data_to_length(x)
    max_length=max(lengths)
    
    for i in range(0,len(sequences)):
        sequences[i]=sequences[i].values
        if sequences[i].shape[0]<max_length:
            features=sequences[i].shape[1]
            compensate=np.empty((max_length-sequences[i].shape[0],features))
            compensate[:]=np.nan
            sequences[i]=np.concatenate((sequences[i],compensate),axis=0)
    
    sequences=np.float32(sequences)
    
    
    return sequences[:,:,1:]
        
        
# test code
'''
path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\SynData\Rate0.9\PartialData'
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide']) 
type=np.array(['numeric','dummy'])
covariate_types=np.array(['numeric']*len(covariates))
sample_size=50
data,lengths=data_loader(path,sample_size,covariates,covariate_types)   
'''
    
    