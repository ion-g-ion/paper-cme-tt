#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:18:43 2021

@author: yonnss
"""



import tt
import scipy.io
import numpy as np
from CME import CME,Gillespie,CompleteObservations,Observations_grid
import matplotlib.pyplot as plt
import scipy.integrate
import tt.amen
import timeit
import sys
import scipy.interpolate
import scipy.stats
from mpl_toolkits import mplot3d
from ttInt import ttInt
from mcmc import *
import pickle
import datetime

import mkl
mkl.set_num_threads_local(32)

def lagrange (x ,i , xm ):
    """
    Evaluates the i-th Lagrange polynomial at x
    based on grid data xm
    """
    n=len( xm )-1
    y=1 
    for j in range ( n+1 ):
        if i!=j:
            y*=( x-xm[j])/( xm[i]-xm[j])
    return y

def points_weights(a,b,nl):
    pts,ws = np.polynomial.legendre.leggauss(Nl)
    pts = 0.5 * (b-a) * (pts+1) + a
    ws = (b-a) / 2  *ws
    return pts, ws

def gamma_params(mode,var):
    beta = (mode+np.sqrt(mode**2+4*var))/(2*var)
    alpha = mode * beta + 1
    return alpha,beta

rates = np.array([0.002,0.015,0.1,0.01])
Pre =np.array( [[1,0],[1,0],[0,0],[0,1]])
Post = np.array([[0,0],[1,1],[1,0],[0,0]])
Props = [ lambda x: x[:,0], lambda x: x[:,0]  , lambda x: x[:,0]*0+1  , lambda x: x[:,1] ]


# construct the model and the CME operator
N = [64, 64] # state truncation
Initial = [2,4]
mdl_true = CME(N, Pre,Post,rates,Props)
x0 = np.zeros(N)
x0[tuple(Initial)] = 1.0



# Set up model
mdl = CME(N, Pre,Post,rates*0+1,Props)
Atts = mdl.construct_generator_tt(as_list = True)
for i in range(rates.size):
    pass

No = 64
dT = 4
time_observation = np.arange(No)*dT


#%% Get observation
np.random.seed(12345)

time_observation = np.arange(No)*dT
sigma = 1/2
reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(Initial),time_observation[-1],Pre,Post-Pre,rates)
observations = Observations_grid(time_observation, reaction_time, reaction_jumps)
observations_noise = observations+np.random.normal(0,sigma,observations.shape)

plt.figure()
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,0],2)[:-1],'b')
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,1],2)[:-1],'r') 
plt.scatter(time_observation,observations_noise[:,0],c='b',s=20)
plt.scatter(time_observation,observations_noise[:,1],c='r',marker='x',s=20)
plt.pause(0.05)
plt.xlabel('t [s]')
plt.ylabel('#individuals')
plt.legend(['mRNA','Protein','mRNA observation','Protein observation'])

dct = {'time_observation' : time_observation, 'observations' : observations, 'observations_noise': observations_noise, 'sigma': sigma, 'reaction_time' : reaction_time, 'reaction_jumps' : reaction_jumps ,'reaction_indices' : reaction_indices}
with open('simplegene_64_500k.pickle', 'wb') as handle:
    pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

gamma_pdf = lambda x,a,b : b**a * x**(a-1) * np.exp(-b*x) / scipy.special.gamma(a)
lognormal_pdf = lambda x,m,s : np.exp(-(np.log(x)-m)**2*0.5/s/s)/(x*s*np.sqrt(2*np.pi))

def ObservationOperator(y,P,t):
    sigma = 1/2
    PO = tt.kron(tt.tensor(np.exp(-0.5*(y[0]-np.arange(N[0]))**2/sigma**2)),tt.tensor(np.exp(-0.5*(y[1]-np.arange(N[1]))**2/sigma**2))) *(1/(sigma**2*2*np.pi))
    # PO = PO * (1/tt.sum(PO))
    return PO


#%% Prior
# IC
P0 = tt.tensor(x0)
# Prior 
mu = rates
var = rates / np.array([1000,350,25,600])
alpha_prior = mu**2/var
beta_prior = mu/var
# Pt = tt.kron( tt.kron( tt.tensor( gamma_pdf(pts1,alpha_prior[0],beta_prior[0]) ) , tt.tensor( gamma_pdf(pts2,alpha_prior[1],beta_prior[1]) ) ),tt.kron( tt.tensor( gamma_pdf(pts3,alpha_prior[2],beta_prior[2]) ) , tt.tensor( gamma_pdf(pts4,alpha_prior[3],beta_prior[3]) ) ) )


params = []

Ns = 500000

param_now = np.random.gamma(alpha_prior,1/beta_prior)
# Pprev = eval_post(Atts,param_now,P0,time_observation,observations_noise,ObservationOperator,eps=1e-5,method = 'crank–nicolson',dtmax=2,Nmax = 64)
Pprev = eval_post_full(mdl,param_now,P0,time_observation,observations_noise,ObservationOperator,eps=1e-5,method = 'crank–nicolson',dtmax=2,Nmax = 64)

Pprev = np.prod(Pprev) * np.prod(gamma_pdf(param_now, alpha_prior, beta_prior))
params.append(param_now)

total_sampled = 0
time_start = datetime.datetime.now()
while len(params)<Ns:
    total_sampled += 1
    
    
    tme_iteration = datetime.datetime.now()
    param_new = np.random.lognormal(np.log(param_now)-0.01/2,0.1)
    gnew = np.prod(lognormal_pdf(param_new,np.log(param_now)-0.01/2,0.1))
    
    gnewinv = np.prod(lognormal_pdf(param_now,np.log(param_new)-0.01/2,0.1))
    
    print('new proposal ',param_new,' ground truth ',rates)
    # Pnew = eval_post(Atts,param_new,P0,time_observation,observations_noise,ObservationOperator,eps=1e-5,method = 'crank–nicolson',dtmax=2,Nmax = 64)
    Pnew = eval_post_full(mdl,param_new,P0,time_observation,observations_noise,ObservationOperator,eps=1e-5,method = 'crank–nicolson',dtmax=2,Nmax = 64)
    
    Pnew = np.prod(Pnew) * np.prod(gamma_pdf(param_new, alpha_prior, beta_prior))

    acceptance = np.min([1,Pnew/Pprev*gnewinv/gnew])
    print('\tacceptance ratio ',acceptance)
    number = np.random.rand()
    
    if number < acceptance:
        param_now = param_new.copy()
        params.append(param_now.copy())
        
        Pprev = Pnew
    else:
        pass
    print('\taccepted ',len(params),'/',total_sampled,' percentage ',len(params)/total_sampled*100)
    time_remaining = Ns * (datetime.datetime.now()-time_start) / len(params) - (datetime.datetime.now()-time_start)
    print('\telapsed time ',datetime.datetime.now()-time_start,' remaining',time_remaining)
    tme_iteration = datetime.datetime.now() - tme_iteration
    print('\ttime',tme_iteration)
    
params = np.array(params)

dct = {'time_observation' : time_observation, 'observations' : observations, 'observations_noise': observations_noise, 'sigma': sigma, 'reaction_time' : reaction_time, 'reaction_jumps' : reaction_jumps ,'reaction_indices' : reaction_indices, 'sample' : params}
with open('simplegene_64_500k.pickle', 'wb') as handle:
    pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%

nburn = 60000
print('Mean ', np.mean(params[nburn:,:],0))
print('std  ', np.std(params[nburn:,:],0))
print('var  ', np.std(params[nburn:,:],0)**2)

plt.figure()
plt.hist(params[nburn:,0],bins = 100)
plt.axvline(rates[0],c='r',linestyle=':')

plt.figure()
plt.hist(params[nburn:,1],bins = 100)
plt.axvline(rates[1],c='r',linestyle=':')

plt.figure()
plt.hist(params[nburn:,2],bins = 100)
plt.axvline(rates[2],c='r',linestyle=':')

plt.figure()
plt.hist(params[nburn:,3],bins = 100)
plt.axvline(rates[3],c='r',linestyle=':')
