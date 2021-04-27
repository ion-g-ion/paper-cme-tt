#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:50:05 2021

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



# species are G,M,P,G*
rates = np.array([4.0,10.0,1.0,0.2,0.6,1.0])
Pre =np.array( [[1,0,0,0],[0,1,0,0],[0,1,0,0],[1,0,1,0],[0,0,0,1],[0,0,1,0]])
Post = np.array([[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,1],[1,0,1,0],[0,0,0,0]])
Props = [ lambda x: x[:,0], lambda x: x[:,1]  , lambda x: x[:,1]  , lambda x: x[:,0]*x[:,2] , lambda x: x[:,3], lambda x: x[:,2] ]


# construct the model and the CME operator
N = [4, 32, 128 ,4] # state truncation
# N = [2, 16, 64 ,2] # state truncation
Initial = [1,0,0,0]
mdl_true = CME(N, Pre,Post,rates,Props)
Att_true = mdl_true.construct_generator_tt()
x0 = np.zeros(N)
x0[tuple(Initial)] = 1.0

qtt = True


# Set up model
mdl = CME(N, Pre,Post,rates*0+1,Props)
Atts = mdl.construct_generator_tt(as_list = True)

Nl = 64
mult = 5
param_range = [[r/1000,r*mult] for r in rates[:-1]]



# basis = [LegendreBasis(Nl,[p[0],p[1]]) for p in param_range]
basis = [BSplineBasis(Nl,[p[0],p[1]],deg = 3) for p in param_range]

pts = [b.integration_points(4)[0] for b in basis]
ws  = [b.integration_points(4)[1] for b in basis]
lint = pts[0].size

WS = tt.mkron([tt.tensor(b.get_integral()) for b in basis])


A_tt = extend_cme(Atts,pts+[np.array([rates[-1]])])
A_tt = A_tt.round(1e-10,20)

mass_tt,mass_inv_tt = get_mass(basis)
stiff_tt = get_stiff(A_tt,N,pts,ws,basis)
M_tt = tt.kron(tt.eye(N),mass_inv_tt) @ stiff_tt


#%% Get observation
np.random.seed(34548)


# reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(Initial),time_observation[-1],Pre,Post-Pre,rates)
# observations = Observations_grid(time_observation, reaction_time, reaction_jumps)
# observations_noise = observations+np.random.normal(0,sigma,observations.shape)

with open(r"3stage2_45_500k.pickle", "rb") as input_file:
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

plt.figure()
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,0],2)[:-1],'b')
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,1],2)[:-1],'r') 
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,2],2)[:-1],'g')
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,3],2)[:-1],'c') 
plt.scatter(time_observation,observations_noise[:,0],c='k',marker='x',s=20)
plt.scatter(time_observation,observations_noise[:,1],c='k',marker='x',s=20)
plt.scatter(time_observation,observations_noise[:,2],c='k',marker='x',s=20)
plt.scatter(time_observation,observations_noise[:,3],c='k',marker='x',s=20)
plt.xlabel('t [s]')
plt.ylabel('#individuals')
plt.legend(['G','M','P','G*','observations'])
tikzplotlib.save('./../results/3stage_45_sample.tex')
plt.pause(0.05)


#%% Observation operator
obs_operator = IndependentGaussianObservation([sigma]*4, N)


#%% Prior and IC
# IC
P = tt.tensor(x0)
# Prior 
mu = rates[:-1]*np.array([1.5,1.5,1.5,1.0,1.0])
var = rates[:-1] * np.array([4/3, 5, 0.25, 0.04, 0.2])
alpha_prior = mu**2/var
beta_prior = mu/var
Prior1 = GammaPDF(alpha_prior[0], beta_prior[0], basis[0], param_range[0][0], param_range[0][1])
Prior2 = GammaPDF(alpha_prior[1], beta_prior[1], basis[1], param_range[1][0], param_range[1][1])
Prior3 = GammaPDF(alpha_prior[2], beta_prior[2], basis[2], param_range[2][0], param_range[2][1])
Prior4 = GammaPDF(alpha_prior[3], beta_prior[3], basis[3], param_range[3][0], param_range[3][1])
Prior5 = GammaPDF(alpha_prior[4], beta_prior[4], basis[4], param_range[4][0], param_range[4][1])
Priors = [Prior1 , Prior2 , Prior3 , Prior4 , Prior5]
Prior = Prior1 ** Prior2 ** Prior3 ** Prior4 ** Prior5
Puniform = UniformPDF(basis, param_range)

print('Initial E ',Prior.expected_value())
print('Initial C ',Prior.covariance_matrix())

P = tt.kron(P,Prior.tt)


P = P * (1/tt.sum(P * tt.kron(tt.ones(N),WS)))
Post = pdfTT(basis, param_range)

for i in range(5):
    plm = Prior.marginal(np.eye(5)[i,:])
    x = np.linspace(plm.basis[0].domain[0],plm.basis[0].domain[1],1000)
    B = plm.basis[0](x)
    
    plt.figure()
    plt.plot(x,np.einsum('ij,i->j',B,plm.tt.full()))


plt.pause(0.05)
#%% integrator 

if qtt:
    A_qtt = ttm2qttm(M_tt)
    fwd_int = ttInt(A_qtt, epsilon = 1e-5, N_max = 8, dt_max = 1.0,method='cheby')
    ws_qtt = tt2qtt(WS)
    Nbs = 12
    P = tt2qtt(P)
else:
    fwd_int = ttInt(M_tt, epsilon = 1e-5, N_max = 64, dt_max = 1.0,method='crank–nicolson')
    Nbs = 8
    
# import sys
# sys.exit()
Pts = []
Pjs_fwd = []
print('Starting...')
tme_total = datetime.datetime.now()
tensor_size = 0
for i in range(1,No):
    
    y = observations_noise[i,:]

    
    PO = obs_operator(y) 
    
    
    # PO = tt.kron(PO,Puniform.tt)
    PO = tt.kron(PO,tt.ones([Nl]*5))
    # PO = tt.ones(N+5*[Nl])
    if qtt: PO = tt2qtt(PO)
    
    print('new observation ',i,'/',No,' at time ',time_observation[i],' ',y)
    
    tme = datetime.datetime.now()
    P = fwd_int.solve(P, dT, intervals = Nbs,qtt = qtt,verb = False)
    tme = datetime.datetime.now() - tme
    
    
    print('\tmax rank ',max(P.r))
    Ppred = P
    Ppost = PO * Ppred
    Ppost = Ppost.round(1e-10)
    print('\tmax rank (after observation) ',max(Ppost.r))
    
    if tensor_size<tt_size(Ppost): tensor_size = tt_size(Ppost)
    
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
    
    
    if qtt: Pt = qtt2tt(Pt,[Nl]*5)
    Pts.append(Pt.copy())
    
    Post.update(Pt)
    
    E = Post.expected_value()
    
    print('\tExpected value computed posterior ' ,E)
    # print('\tVariance computed posterior       ' ,V)

    P = Ppost
    print('\tposterior size ',sum([elem.size for elem in P.to_list(P)])*8 / 1000000,' MB')
    print('\telapsed ',tme)
    
    # P12 = 




Pt_fwd = Pt
Post.normalize()
tme_total = datetime.datetime.now() - tme_total

#%% Visualize an show info
print('Total time ',tme_total)
print('Max size ',tensor_size*8/1e6)

nburn = 150000
E = Post.expected_value()
C = Post.covariance_matrix()
V = np.diag(C)
print()
print('Exact rates:                      ',rates)
print('')
print('Expected value computed posterior ' ,E)
print('Variance computed posterior       ' ,V)
# print('Computed modes:                   ',theta_mode)
print('')
print('Expected MCMC posterior           ' ,np.mean(sample_posterior_mcmc[nburn:,:],0))
print('Variance MCMC posterior           ' ,np.std(sample_posterior_mcmc[nburn:,:],0))
print('')
print('Relative absolute error exp       ',np.abs(np.mean(sample_posterior_mcmc[nburn:,:],0)-E)/E * 100,' %')
print('Relative absolute error var       ',np.abs(np.std(sample_posterior_mcmc[nburn:,:],0)**2-V)/V * 100,' %')
print('')
print('Expected value prior              ' ,alpha_prior/beta_prior)
print('Variance computed prior           ' ,alpha_prior/beta_prior/beta_prior)
print('')

Post.normalize()
Post1 = Post.marginal(np.eye(5)[0,:])
Post2 = Post.marginal(np.eye(5)[1,:])
Post3 = Post.marginal(np.eye(5)[2,:])
Post4 = Post.marginal(np.eye(5)[3,:])
Post5 = Post.marginal(np.eye(5)[4,:])


for i in range(5):
    prior = Prior.marginal(np.eye(5)[i,:])
    post = Post.marginal(np.eye(5)[i,:])
    x = np.linspace(prior.basis[0].domain[0],prior.basis[0].domain[1],1000)
    B = prior.basis[0](x)
    
    plt.figure()
    plt.plot(x,np.einsum('ij,i->j',B,post.tt.full()))
    plt.hist(sample_posterior_mcmc[nburn:,i],bins=128,density=True,color='c',alpha=0.4)
    plt.axvline(rates[i],c='r',linestyle=':')
    plt.plot(x,np.einsum('ij,i->j',B,prior.tt.full()),'g:')


plt.figure()

k = 0
for i in range(5):
    for j in range(5):
        k += 1
        if i==j:
            plt.subplot(5, 5, k)
            
            prior = Prior.marginal(np.eye(5)[i,:])
            post = Post.marginal(np.eye(5)[i,:])
            theta = np.linspace(prior.basis[0].domain[0],prior.basis[0].domain[1],1000)
            B = prior.basis[0](theta)
            
            post = np.einsum('ij,i->j',B,post.tt.full())
            prior = np.einsum('ij,i->j',B,prior.tt.full())
            plt.plot(theta,post/np.max(post)*np.max(theta))
            count, bins = np.histogram(sample_posterior_mcmc[nburn:,i],bins=128,density=True)
            count = count/np.max(post)*np.max(theta)
            plt.hist(bins[:-1], bins, weights=count,color='c',alpha=0.4)
            plt.axvline(rates[i],c='r',linestyle=':')
            plt.plot(theta,prior/np.max(post)*np.max(theta),'g:')
            
        else:
            plt.subplot(5, 5, k)
            
            post = Post.marginal(np.eye(5)[i,:]+np.eye(5)[j,:])
            t1, t2 = np.meshgrid(np.linspace(param_range[i][0],param_range[i][1],128),np.linspace(param_range[j][0],param_range[j][1],128))
            B1 = basis[i](np.linspace(param_range[i][0],param_range[i][1],128))
            B2 = basis[j](np.linspace(param_range[j][0],param_range[j][1],128))
            post = post.tt.full()
            post = np.einsum('ij,im,jn->mn',post,B1,B2)
            plt.contourf(t2,t1, post, cmap='gray_r', levels =32)
            plt.axvline(rates[j],c='r',linestyle=':',linewidth=1)
            plt.axhline(rates[i],c='r',linestyle=':',linewidth=1)
            
        
        if i==4: plt.xlabel(r'$\theta_'+str(j+1)+'$')
        if j==0: plt.ylabel(r'$\theta_'+str(i+1)+'$')
        
        if j>0: plt.yticks([])
        if i<4: plt.xticks([])