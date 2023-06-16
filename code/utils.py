# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:22:13 2022

@author: s1155151972
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import Adv_SD_generator as sdg
import Adv_model as Model
import pandas as pd
import multiprocessing as mp
import time
import os
import argparse
import utils
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# find the permutation that minimizes the distance between pi and estimated pi
def find_permute(est_pi,pi):
    '''
    Parameters
    ----------
    est_pi : Numpy Array (1,n)
        The estimated pi
    pi : Numpy Array (1,n)
        True pi

    Returns
    -------
    list
        Return the permutation that maps estimated pi to the true pi

    '''
    est_pi=np.array(est_pi)
    pi=np.array(pi)
    index=[i for i in range(0,len(pi))]
    all_permutations=list(itertools.permutations(index))
    # calcualte distance
    distance=[]
    for i in range(0,len(all_permutations)):
        pmt=list(all_permutations[i])
        distance.append(np.linalg.norm(est_pi[pmt]-pi))
    return list(all_permutations[np.argmin(distance)])

# permute training history of A, B and pi
def permute_train(permute,est_pi,est_alpha,est_mu,est_sigma,est_beta):
    '''
    permute: a list used to permute
    est_A: the simulated transition matrix, shape = (iter, z,z)
    est_pi: the simulated initial distribution, shape=(iter,z)
    est_mu: the simulated mean for x, shape = (iter,z,x)
    est_sigma: the simulated covar for x, shape = (iter, z,x,x)
    est_beta: the simulated covariates for x, shape = (iter,z,y,x)
    '''
    est_alpha=np.array(est_alpha)
    est_pi=np.array(est_pi)
    est_mu=np.array(est_mu)
    est_sigma=np.array(est_sigma)
    est_beta=np.array(est_beta)
    
    for i in range(0,len(est_alpha)):
        # permute est_A
        est_alpha[i]=est_alpha[i][permute,:,:]
        est_alpha[i]=est_alpha[i][:,permute,:]
        # permute est_pi
        est_pi[i]=est_pi[i][permute]
        # permute mu and sigma
        est_mu[i]=est_mu[i][permute,:]
        est_sigma[i]=est_sigma[i][permute,:,:]
        # permute beta
        est_beta[i]=est_beta[i][permute,:]
        
        
    return est_pi,est_alpha,est_mu,est_sigma,est_beta


def make_category(parent_name, save_paths):
    '''
    make a category to store simulation results
    parent_name: the name of the parent category where all resutls are stores
    the file should be stored in 'parent_name/save_paths[i]'
    return concatenation of parent_name and save_paths
    '''
    os.mkdir(parent_name)
    names=[]
    for i in range(len(save_paths)):
        path=save_paths[i]
        os.mkdir(f'{parent_name}/{path}')
        names.append(f'{parent_name}/{path}')
    return names


# simulation function
def inference(path,sample_size,covariates,covariate_types,args,iterations,prob):
    '''
    path: data path
    sample_size: sample size
    args: args of optimizer
    iterations: total number of iterations
    prob: prob for sampling theta (parameter)
    '''
    data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types) 
    m=Model.HMM_Model(data,lengths,covariates,covariate_types)
    optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
    param=optimizer.run(n=iterations,log_step=4,prog_bar=True,prob=prob,initial_x=None,initial_z=None)
    
    return optimizer

def simulation(paths,param_path,param_names,sample_size,covariates,covariate_types,args,iterations,nums,prob,opts=None):
    '''
    paths: list of path
    nums: number of simulations on each dataset
    param_path: path for parameters
    param_names: names of parameter
    '''
    '''
    opts=[]
    for i in range(len(paths)):
        path=paths[i]
        opts.append([])
        for j in range(nums):
            optimizer=inference(path,sample_size,covariates,covariate_types,args,iterations,prob)
            opts[i].append(optimizer)
    '''
    # read true parameters
    initial=np.load(f'{param_path}\initial.npy')
    transition=np.load(f'{param_path}/transition.npy')
    beta=np.load(f'{param_path}/beta.npy')
    mu=np.load(f'{param_path}\mu.npy')
    sigma=np.load(f'{param_path}\sigma.npy')
    
    # map estimated parameters to true parameters
    for i in range(len(opts)):
        for j in range(len(opts[i])):
            ep=opts[i][j].param['pi'][-1]
            permute=find_permute(ep,initial)
            p,t,m,s,b=permute_train(permute,opts[i][j].param['pi'],opts[i][j].param['transition'],
                                          opts[i][j].param['mu'],opts[i][j].param['sigma'],opts[i][j].param['beta'])
            opts[i][j].param['pi']=p
            opts[i][j].param['transition']=t
            opts[i][j].param['mu']=m
            opts[i][j].param['sigma']=s
            opts[i][j].param['beta']=b
    
    return opts

def generate_plot(opts, paths,param_path):
    '''
    generate trace plots
    opts: the 2D list of optimizers
    paths: the list of paths to save the plots
    param_names: names of parameter
    '''
    
    # read true parameters
    initial=np.load(f'{param_path}\initial.npy')
    transition=np.load(f'{param_path}/transition.npy')
    beta=np.load(f'{param_path}/beta.npy')
    mu=np.load(f'{param_path}\mu.npy')
    sigma=np.load(f'{param_path}\sigma.npy')
    
    count=1
    # plot pi
    for i in range(len(opts)):
        plt.subplots(len(opts[i]),len(initial))
        for j in range(len(opts[i])):
            # plot pi
            plt.subplot(len(opts[i]),len(initial),count)
            opt=opts[i][j]
            plt.plot(opt.param['pi'][:,j])
        plt.savefig(f'{paths[i]}\pi.png')


def pickle_data(opts,save_paths):
    '''
    save data of opts objects to paths
    opts: a 2d list of optimizer objects
    save_path: paths to save to
    '''
    paths=save_paths
    for i in range(len(opts)):
        path=paths[i]
        for j in range(len(opts[i])):
            name=f'{path}/Simulation{j+1}'
            os.mkdir(name)
            optimizer=opts[i][j]
            optimizer.pickle(name)


def unpickle_data(model,paths,nums):
    '''
    model: a hmm model
    paths of files
    nums of simulation in each path
    '''
    f=len(paths)
    opts=[[Model.Random_Gibbs(model) for j in range(nums)] for i in range(f)]
    for i in range(len(paths)):
        path=paths[i]
        files=os.listdir(path)
        for j in range(nums):
            param={}
            tmp_path=f'{path}/{files[j]}'
            opts[i][j].unpicle(tmp_path)
    return opts
            
            
            
        
            

