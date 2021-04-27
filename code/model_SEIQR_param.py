#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:13:57 2021

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
from tt_aux import *
import datetime
from pdfTT import *
import ObservationOperators 
# speices S,E,I,Q,R

AV = 800
rates = np.array([0.04,0.4,0.4,0.004,0.12,0.8765,0.01,0.01,0.01])
# 0.1,0.5,1.0,0.01,0.01,0.1,0.4
Pre =[[1,0,1,0,0]]    # S+I->E+I
Pre.append( [0,1,0,0,0])    # E->I
Pre.append( [0,0,1,0,0])    # I->Q
Pre.append( [0,0,1,0,0])     # I->none
Pre.append( [0,0,1,0,0])    # I->R
Pre.append( [0,0,0,1,0])     # Q-> R
Pre.append( [0,0,1,0,0])    # I->S
Pre.append( [0,0,0,1,0])     # Q->S
Pre.append( [0,0,0,0,0])     # none ->S
Pre = np.array(Pre)

Post = [[0,1,1,0,0]]
Post.append([0,0,1,0,0])
Post.append([0,0,0,1,0])
Post.append([0,0,0,0,0])
Post.append([0,0,0,0,1])
Post.append([0,0,0,0,1])
Post.append([1,0,0,0,0])
Post.append([1,0,0,0,0])
Post.append([1,0,0,0,0])
Post = np.array(Post)       
    
Props = [ lambda x: x[:,0]*x[:,2] ]
Props.append(lambda x: x[:,1])
Props.append(lambda x: x[:,2])
Props.append(lambda x: x[:,2])
Props.append(lambda x: x[:,2])
Props.append(lambda x: x[:,3])
Props.append(lambda x: x[:,2])
Props.append(lambda x: x[:,3])
Props.append(lambda x: x[:,1]*0+1)

         



# construct the model and the CME operator
N = [128]*5 # state truncation
N = [128,64,64,32,32] # state truncation
mdl_true = CME(N, Pre,Post,rates,Props)
mdl = CME(N, Pre,Post,rates*0+1,Props)
s0 = [90,0,4,0,0]
Tend = 5
qtt = True
Att = mdl_true.construct_generator_tt(as_list = False)
Atts = mdl.construct_generator_tt(as_list = True)
No = 45
time_observation = np.linspace(0,Tend,No+1)

#%% parameter dependent master operator
Nl = 64
mult = 5
param_range = [[0,r*mult] for r in rates[:4]]


# basis = [LegendreBasis(Nl,[p[0],p[1]]) for p in param_range]
basis = [BSplineBasis(Nl,[p[0],p[1]],deg = 2) for p in param_range]

pts = [b.integration_points(4)[0] for b in basis]
ws  = [b.integration_points(4)[1] for b in basis]

WS = tt.mkron([tt.tensor(b.get_integral()) for b in basis])

A_tt = extend_cme(Atts,pts+[np.array([r]) for r in rates[4:]])
A_tt = A_tt.round(1e-10,20)


mass_tt,mass_inv_tt = get_mass(basis)
stiff_tt = get_stiff(A_tt,N,pts,ws,basis)
M_tt = tt.kron(tt.eye(N),mass_inv_tt) @ stiff_tt

#%% Observation 
sigmas = [0.1,0.1,0.1,0.01,0.01]
obs_operator = ObservationOperators.IndependentLogNormalObservation(sigmas, N)

np.random.seed(13212)
reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(s0),Tend,Pre,Post-Pre,rates)
observations = Observations_grid(time_observation, reaction_time, reaction_jumps)
observations_noise = obs_operator.add_noise(observations)

# plot the path
plt.figure()
for i in range(5): plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,i],2)[:-1])
for i in range(5): plt.scatter(time_observation,observations_noise[:,i],marker='x')
plt.legend(['S','E','I','Q','R'])
plt.xlabel(r'$t$')
plt.ylabel(r'#individuals')
import tikzplotlib
tikzplotlib.save('../results/seiqr_sample2.tikz')
plt.pause(0.05)

#%% Prior

mu = rates[:4]*np.array([1.5,1.5,1.5,1.0])
var = rates[:4] * np.array([0.025, 0.1, 0.25, 0.0001])
alpha_prior = mu**2/var
beta_prior = mu/var
Priors = [GammaPDF(alpha_prior[i], beta_prior[i], basis[i], param_range[i][0], param_range[i][1]) for i in range(4)]
Prior = Priors[0] ** Priors[1] ** Priors[2] ** Priors[3] 
Puniform = UniformPDF(basis, param_range)

for i in range(4):
    plm = Prior.marginal(np.eye(4)[i,:])
    x = np.linspace(plm.basis[0].domain[0],plm.basis[0].domain[1],1000)
    B = plm.basis[0](x)
    plt.figure()
    plt.plot(x,np.einsum('ij,i->j',B,plm.tt.full()))
    

P0 = SingularPMF(N,s0)

P = tt.kron(P0,Prior.tt)
P = P * (1/tt.sum(P * tt.kron(tt.ones(N),WS)))
Post = pdfTT(basis, param_range)


#%% Loop

if qtt:
    A_qtt = ttm2qttm(M_tt)
    fwd_int = ttInt(A_qtt, epsilon = 1e-5, N_max = 8, dt_max = 1.0,method='cheby')
    ws_qtt = tt2qtt(WS)
    Nbs = 1
    P = tt2qtt(P)
else:
    fwd_int = ttInt(M_tt, epsilon = 1e-5, N_max = 64, dt_max = 1.0,method='crank–nicolson')
    Nbs = 1
    
# import sys
# sys.exit()
Pts = []
print('Starting...')
tme_total = datetime.datetime.now()
tensor_size = 0
for i in range(1,No):
    
    y = observations_noise[i,:]
    dT = time_observation[i]-time_observation[i-1]
    
    PO = obs_operator(y) 
    
    PO = tt.kron(PO,tt.ones([Nl]*4))
    if qtt: PO = tt2qtt(PO)
    
    print('new observation ',i,'/',No,' at time ',time_observation[i],' ',y,flush=True)
    
    tme = datetime.datetime.now()
    P = fwd_int.solve(P, dT, intervals = Nbs,qtt = qtt,verb = False,rounding=True)
    tme = datetime.datetime.now() - tme
    print('',flush=True)
    
    print('\tmax rank ',max(P.r))
    Ppred = P
    Ppost = PO * Ppred
    Ppost = Ppost.round(1e-10)
    print('\tmax rank (after observation) ',max(Ppost.r))
    
    if tensor_size<tt_size(Ppost): tensor_size = tt_size(Ppost)
    
    if not qtt:
        # Ppost = Ppost * (1/tt.sum(Ppost * tt.kron(tt.ones(N),WS)))
        Pt = tt.sum(tt.sum(tt.sum(tt.sum(Ppost,0),0),0),0) 
        
        Z = tt.dot(Pt,WS)
        Pt = Pt * (1/Z)
        Pt = Pt.round(1e-10)
    
    else:
        # Ppost = Ppost * (1/tt.sum(Ppost * tt.kron(tt.ones(int(np.sum(np.log2(N)))*[2]),ws_qtt)))
        Pt = Ppost
        for i in range(int(np.sum(np.log2(N)))): Pt = tt.sum(Pt,0) 
        Z = tt.dot(Pt,ws_qtt)
        Pt = Pt * (1/Z)
        Pt = Pt.round(1e-10)
    

    Ppost = Ppost*(1/tt.sum(Ppost))
    
    
    if qtt: Pt = qtt2tt(Pt,[Nl]*4)
    Pts.append(Pt.copy())
    
    Post.update(Pt)
    
    E = Post.expected_value()
    
    print('\tExpected value computed posterior ' ,E)
    # print('\tVariance computed posterior       ' ,V)

    P = Ppost
    print('\tposterior size ',sum([elem.size for elem in P.to_list(P)])*8 / 1000000,' MB')
    print('\telapsed ',tme)
    print('',flush=True)
    # P12 = 

tme_total = datetime.datetime.now() - tme_total 


#%% Visualize an show info
print('Total time ',tme_total)
print('Max size ',tensor_size*8/1e6, ' MB')

import pyswarm
theta_mode, _ = pyswarm.pso(lambda x: -Post(x), np.array(param_range)[:,0], np.array(param_range)[:,1])

nburn = 50000
E = Post.expected_value()
C = Post.covariance_matrix()
V = np.diag(C)
print()
print('Exact rates:                      ',rates)
print('')
print('Expected value computed posterior ' ,E)
print('Variance computed posterior       ' ,V)
print('Computed modes:                   ',theta_mode)
print('')
# print('Expected MCMC posterior           ' ,np.mean(sample_posterior_mcmc[nburn:,:],0))
# print('Variance MCMC posterior           ' ,np.std(sample_posterior_mcmc[nburn:,:],0)**2)
# print('')
# print('Relative absolute error exp       ',np.abs(np.mean(sample_posterior_mcmc[nburn:,:],0)-E)/E * 100,' %')
# print('Relative absolute error var       ',np.abs(np.std(sample_posterior_mcmc[nburn:,:],0)**2-V)/V * 100,' %')
print('')
print('Expected value prior              ' ,alpha_prior/beta_prior)
print('Variance computed prior           ' ,alpha_prior/beta_prior/beta_prior)
print('')

Post.normalize()
Post1 = Post.marginal(np.eye(4)[0,:])
Post2 = Post.marginal(np.eye(4)[1,:])
Post3 = Post.marginal(np.eye(4)[2,:])
Post4 = Post.marginal(np.eye(4)[3,:])


for i in range(4):
    prior = Prior.marginal(np.eye(4)[i,:])
    post = Post.marginal(np.eye(4)[i,:])
    x = np.linspace(prior.basis[0].domain[0],prior.basis[0].domain[1],1000)
    B = prior.basis[0](x)
    
    plt.figure()
    plt.plot(x,np.einsum('ij,i->j',B,post.tt.full()))
    # plt.hist(sample_posterior_mcmc[nburn:,i],bins=128,density=True,color='c',alpha=0.4)
    plt.axvline(rates[i],c='r',linestyle=':')
    plt.plot(x,np.einsum('ij,i->j',B,prior.tt.full()),'g:')


plt.figure()

k = 0
for i in range(4):
    for j in range(4):
        k += 1
        if i==j:
            plt.subplot(4, 4, k)
            
            prior = Prior.marginal(np.eye(4)[i,:])
            post = Post.marginal(np.eye(4)[i,:])
            theta = np.linspace(prior.basis[0].domain[0],prior.basis[0].domain[1],1000)
            B = prior.basis[0](theta)
            
            post = np.einsum('ij,i->j',B,post.tt.full())
            prior = np.einsum('ij,i->j',B,prior.tt.full())
            plt.plot(theta,post/np.max(post)*np.max(theta))
            # count, bins = np.histogram(sample_posterior_mcmc[nburn:,i],bins=128,density=True)
            # count = count/np.max(post)*np.max(theta)
            # plt.hist(bins[:-1], bins, weights=count,color='c',alpha=0.4)
            plt.axvline(rates[i],c='r',linestyle=':')
            plt.plot(theta,prior/np.max(post)*np.max(theta),'g:')
            
        else:
            plt.subplot(4, 4, k)
            
            post = Post.marginal(np.eye(5)[i,:]+np.eye(5)[j,:])
            t1, t2 = np.meshgrid(np.linspace(param_range[i][0],param_range[i][1],128),np.linspace(param_range[j][0],param_range[j][1],128))
            B1 = basis[i](np.linspace(param_range[i][0],param_range[i][1],128))
            B2 = basis[j](np.linspace(param_range[j][0],param_range[j][1],128))
            post = post.tt.full()
            post = np.einsum('ij,im,jn->mn',post,B1,B2)
            plt.contourf(t2,t1, post, cmap='gray_r', levels =32)
            plt.axvline(rates[j],c='r',linestyle=':',linewidth=1)
            plt.axhline(rates[i],c='r',linestyle=':',linewidth=1)
            
        
        if i==3: plt.xlabel(r'$\theta_'+str(j+1)+'$')
        if j==0: plt.ylabel(r'$\theta_'+str(i+1)+'$')
        
        if j>0: plt.yticks([])
        if i<3: plt.xticks([])
