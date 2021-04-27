#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:52:13 2021

@author: ion
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
import datetime
from tt_aux import *
import pickle
import tikzplotlib
from basis import *
from pdfTT import *
from ObservationOperators import *



# species are mRNA,P
rates = np.array([0.002,0.015,0.1,0.01])
Pre =np.array( [[1,0],[1,0],[0,0],[0,1]])
Post = np.array([[0,0],[1,1],[1,0],[0,0]])
Props = [ lambda x: x[:,0], lambda x: x[:,0]  , lambda x: x[:,0]*0+1  , lambda x: x[:,1] ]


# construct the model and the CME operator
N = [128, 128] # state truncation
# N = [2, 16, 64 ,2] # state truncation
Initial = [2,4]

qtt = True

# Set up model
mdl = CME(N, Pre,Post,rates*0+1,Props)
Atts = mdl.construct_generator_tt(as_list = True)




#%% Get observation
np.random.seed(34548)


# reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(Initial),time_observation[-1],Pre,Post-Pre,rates)
# observations = Observations_grid(time_observation, reaction_time, reaction_jumps)
# observations_noise = observations+np.random.normal(0,sigma,observations.shape)

with open(r"simplegene_64_500k.pickle", "rb") as input_file:
    dct = pickle.load(input_file) 

No = dct['time_observation'].size
time_observation = dct['time_observation']
reaction_time = dct['reaction_time']
reaction_jumps = dct['reaction_jumps']
reaction_indices = dct['reaction_indices']
observations = dct['observations']
observations_noise = dct['observations_noise']
dT = time_observation[1]-time_observation[0]
sigma = dct['sigma']
sample_posterior_mcmc = dct['sample']


#%% Observation operator
obs_operator = IndependentGaussianObservation([sigma]*2, N)

#%% Prior 
# Prior 
mu = rates
var = rates / np.array([1000,350,25,600])
alpha_prior = mu**2/var
beta_prior = mu/var


def identify_params(Nl,range_mult,eps_solver=1e-5,ntmax=8,deg = 2,qtt = True):


    
    
    param_range = [[r/1000,r*range_mult] for r in rates]
    
    # basis = [LegendreBasis(Nl,[p[0],p[1]]) for p in param_range]
    basis = [BSplineBasis(Nl,[p[0],p[1]],deg = deg) for p in param_range]
    
    pts = [b.integration_points(4)[0] for b in basis]
    ws  = [b.integration_points(4)[1] for b in basis]
    lint = pts[0].size
    
    WS = tt.mkron([tt.tensor(b.get_integral()) for b in basis])
    
    
    A_tt = extend_cme(Atts.copy(),pts)
    A_tt = A_tt.round(1e-10,20)
    
    mass_tt,mass_inv_tt = get_mass(basis)
    stiff_tt = get_stiff(A_tt,N,pts,ws,basis)
    M_tt = tt.kron(tt.eye(N),mass_inv_tt) @ stiff_tt
    
    # IC
    P = SingularPMF(N, Initial)
    

    Priors = [GammaPDF(alpha_prior[i], beta_prior[i], basis[i], param_range[i][0], param_range[i][1]) for i in range(4) ]
    Prior = Priors[0] ** Priors[1] ** Priors[2] ** Priors[3] 
    Puniform = UniformPDF(basis, param_range)
    
    
    
    P = tt.kron(P,Prior.tt)
    # P = P * (1/tt.sum(P * tt.kron(tt.ones(N),WS)))
    
    Post = pdfTT(basis, param_range)
       
    # integrator 
    
    if qtt:
        
        A_qtt = ttm2qttm(M_tt)
        fwd_int = ttInt(A_qtt, epsilon = eps_solver, N_max = ntmax, dt_max = 1.0,method='cheby')
        ws_qtt = tt2qtt(WS)
        Nbs = 4
        P = tt2qtt(P)
    else:
        fwd_int = ttInt(M_tt, epsilon = eps_solver, N_max = ntmax, dt_max = 1.0,method='crank–nicolson')
        Nbs = 4
        
   
    
    # print('Starting...')
    tme_total = datetime.datetime.now()
    storage = 0
    for i in range(1,No):
        
        y = observations_noise[i,:]
    
        
        PO = obs_operator(y) 
        
        
        # PO = tt.kron(PO,Puniform.tt)
        PO = tt.kron(PO,tt.ones([Nl]*4))
        # PO = tt.ones(N+5*[Nl])
        if qtt: PO = tt2qtt(PO)
        

        P = fwd_int.solve(P, dT, intervals = Nbs,qtt = qtt,verb = False,nswp=100)

        
        Ppred = P
        Ppost = PO * Ppred
        Ppost = Ppost.round(1e-10)
     
        if storage<tt_size(Ppost): storage = tt_size(Ppost)
        
        if not qtt:
            # Ppost = Ppost * (1/tt.sum(Ppost * tt.kron(tt.ones(N),WS)))
            Pt = tt.sum(tt.sum(tt.sum(tt.sum(Ppost,0),0),0),0) 
            
            Z = tt.sum(Pt*WS)
            Pt = Pt * (1/Z)
            Pt = Pt.round(1e-10)
        
        else:
            # Ppost = Ppost * (1/tt.sum(Ppost * tt.kron(tt.ones(int(np.sum(np.log2(N)))*[2]),ws_qtt)))
            Pt = Ppost
            for i in range(int(np.sum(np.log2(N)))): Pt = tt.sum(Pt,0) 
            Z = tt.sum(Pt*ws_qtt)
            Pt = Pt * (1/Z)
            Pt = Pt.round(1e-10)
        
    
        Ppost = Ppost*(1/tt.sum(Ppost))
        
        
        if qtt: Pt = qtt2tt(Pt,[Nl]*4)
        
        
        Post.update(Pt)
            
        P = Ppost
        
        
    
    Post.normalize()
    E = Post.expected_value()
    C = Post.covariance_matrix()
    V = np.diag(C)

    return Post, E, V, C, storage

#%% Sweeping



epsilons = [1e-3,1e-4,1e-5,1e-6]
Nls = [16,32,64,128]
ntmax = [4,8,16,32]
degs = [1,2,3,4]

nburn = 250000
mean = np.mean(sample_posterior_mcmc[nburn:,:],0)
var = np.std(sample_posterior_mcmc[nburn:,:],0)**2

Pt_ref, E_ref, V_ref, C_ref, _ = identify_params(64,6,1e-7,16)

Es1 = []
Vs1 = []
time1 = []
memory1 = []
for eps in epsilons:
    print('epsilon ',eps)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(64,6,eps,8)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es1.append(E)
    Vs1.append(V)
    time1.append(tme)
    memory1.append(mem)

errE1 = np.abs(Es1 - mean)/np.outer(np.ones(4),mean) * 100
errV1 = np.abs(Vs1 - var)/np.outer(np.ones(4),var) * 100
print('error w.r.t. MCMC expectation \n',errE1)
print('error w.r.t. MCMC variance\n',errV1)
errE1 = np.abs(Es1 - E_ref)/np.outer(np.ones(4),mean) 
errV1 = np.abs(Vs1 - V_ref)/np.outer(np.ones(4),var) 
print('error w.r.t. fine posterior expectation\n ',errE1)
print('error w.r.t. fine posterior variance\n',errV1)
print('times ',time1)
print('memory ',memory1)

Es2 = []
Vs2 = []
time2 = []
memory2 = []
for Nl in Nls:
    print('ell ',Nl)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(Nl,6,1e-6,8)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es2.append(E)
    Vs2.append(V)
    time2.append(tme)
    memory2.append(mem)
    
errE2 = np.abs(Es2 - mean)/mean * 100
errV2 = np.abs(Vs2 - var)/var * 100
print('error w.r.t. MCMC expectation ',errE2)
print('error w.r.t. MCMC variance',errV2)
errE2 = np.abs(Es2 - E_ref)/mean * 100
errV2 = np.abs(Vs2 - V_ref)/var * 100
print('error w.r.t. fine posterior expectation ',errE2)
print('error w.r.t. fine posterior variance',errV2)
print('times ',time2)
print('memory ',memory2)

Es3 = []
Vs3 = []
time3 = []
memory3 = []
for nt in ntmax:
    print('T ',nt)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(64,6,1e-6,nt)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es3.append(E)
    Vs3.append(V)
    time3.append(tme)
    memory3.append(mem)
   
errE3 = np.abs(Es3 - mean)/mean * 100
errV3 = np.abs(Vs3 - var)/var * 100
print('error w.r.t. MCMC expectation ',errE3)
print('error w.r.t. MCMC variance',errV3)
errE3 = np.abs(Es3 - E_ref)/mean * 100
errV3 = np.abs(Vs3 - V_ref)/var * 100
print('error w.r.t. fine posterior expectation ',errE3)
print('error w.r.t. fine posterior variance',errV3)
print('times ',time3)
print('memory ',memory3)

Es4 = []
Vs4 = []
time4 = []
memory4 = []
for deg in degs:
    print('deg ',deg)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(64,6,1e-5,8,deg=deg)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es4.append(E)
    Vs4.append(V)
    time4.append(tme)
    memory4.append(mem)
   
errE4 = np.abs(Es4 - mean)/mean * 100
errV4 = np.abs(Vs4 - var)/var * 100
print('error w.r.t. MCMC expectation ',errE4)
print('error w.r.t. MCMC variance',errV4)
errE4 = np.abs(Es4 - E_ref)/mean * 100
errV4 = np.abs(Vs4 - V_ref)/var * 100
print('error w.r.t. fine posterior expectation ',errE4)
print('error w.r.t. fine posterior variance',errV4)
print('times ',time4)
print('memory ',memory4)


#%%

Es1 = np.array(Es1)
Es2 = np.array(Es2)
Es3 = np.array(Es3)

Vs1 = np.array(Vs1)
Vs2 = np.array(Vs2) 
Vs3 = np.array(Vs3)
    
errE1 = np.abs(Es1 - mean)/mean * 100
errE2 = np.abs(Es2 - mean)/mean * 100
errE3 = np.abs(Es3 - mean)/mean * 100

errV1 = np.abs(Vs1 - var)/var * 100
errV2 = np.abs(Vs2 - var)/var * 100
errV3 = np.abs(Vs3 - var)/var * 100