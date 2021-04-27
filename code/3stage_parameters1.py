
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:08:32 2020

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



# species are G,M,P,G*
rates = np.array([4.0,10.0,1.0,0.2,0.6,1.0])
Pre =np.array( [[1,0,0,0],[0,1,0,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
Post = np.array([[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,0]])
Props = [ lambda x: x[:,0], lambda x: x[:,1]  , lambda x: x[:,1]  , lambda x: x[:,0] , lambda x: x[:,3], lambda x: x[:,2] ]


# construct the model and the CME operator
N = [4, 32, 128 ,4] # state truncation
Initial = [1,0,0,0]
mdl_true = CME(N, Pre,Post,rates,Props)
x0 = np.zeros(N)
x0[tuple(Initial)] = 1.0

qtt = True


# Set up model
mdl = CME(N, Pre,Post,rates*0+1,Props)
Atts = mdl.construct_generator_tt(as_list = True)

Nl = 64
mult = 4
param_range = [(0,r*5) for r in rates[:-1]]
pts1, ws1 = points_weights(param_range[0][0],param_range[0][1],Nl)
pts2, ws2 = points_weights(param_range[1][0],param_range[1][1],Nl)
pts3, ws3 = points_weights(param_range[2][0],param_range[2][1],Nl)
pts4, ws4 = points_weights(param_range[3][0],param_range[3][1],Nl)
pts5, ws5 = points_weights(param_range[4][0],param_range[4][1],Nl)

A_tt = tt.kron(Atts[0] , tt.kron(tt.matrix(np.diag(pts1)),tt.eye([Nl]*4)) ) \
     + tt.kron(Atts[1] , tt.kron(tt.kron(tt.eye([Nl]),tt.matrix(np.diag(pts2))),tt.eye([Nl]*3)) ) \
     + tt.kron(Atts[2] , tt.kron(tt.kron(tt.eye([Nl]*2),tt.matrix(np.diag(pts3))),tt.eye([Nl]*2)) )  \
     + tt.kron(Atts[3] , tt.kron(tt.kron(tt.eye([Nl]*3),tt.matrix(np.diag(pts4))),tt.eye([Nl]*1)) ) \
     + tt.kron(Atts[4] , tt.kron(tt.eye([Nl]*4),tt.matrix(np.diag(pts5))) ) \
     + tt.kron(Atts[5], tt.eye([Nl]*5) )*rates[5]

A_tt = A_tt.round(1e-10,20)

No = 45
# Nt = 64
dT = 0.15
Nbs = 4
time_observation = np.arange(No)*dT


#%% Get observation
np.random.seed(7465343)

time_observation = np.arange(No)*dT

sigma = 0.3

# fine_time_grid = np.sort(np.concatenate((np.linspace(0,time_observation[-1],No*500),time_observation)))
reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(Initial),time_observation[-1],Pre,Post-Pre,rates)
observations = Observations_grid(time_observation, reaction_time, reaction_jumps)
# observations_fine = Observations_grid(fine_time_grid, reaction_time, reaction_jumps)
observations_noise = observations+np.random.normal(0,sigma,observations.shape)

plt.figure()
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,0],2)[:-1],'b')
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,1],2)[:-1],'r') 
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,2],2)[:-1],'g')
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,3],2)[:-1],'c') 
plt.scatter(time_observation,observations_noise[:,0],c='k',marker='x',s=20)
plt.scatter(time_observation,observations_noise[:,1],c='k',marker='x',s=20)
plt.scatter(time_observation,observations_noise[:,2],c='k',marker='x',s=20)
plt.scatter(time_observation,observations_noise[:,3],c='k',marker='x',s=20)

plt.pause(0.05)
plt.xlabel('t [s]')
plt.ylabel('#individuals')
plt.legend(['G','M','P','G*'])


gamma_pdf = lambda x,a,b : x**(a-1) * np.exp(-b*x)


#%% Loops
# IC
P = tt.kron(tt.tensor(x0),tt.ones([Nl]*5))
# Prior 
# alpha_prior, beta_prior = gamma_params(rates,rates / np.array([1000,250,25,900]))
mu = rates[:-1]*np.array([1.5,1.5,1.5,1.2,1.2])
var = rates[:-1] * np.array([4/3, 5, 0.25, 0.04, 0.2])
alpha_prior = mu**2/var
beta_prior = mu/var
Pt = tt.tensor( gamma_pdf(pts1,alpha_prior[0],beta_prior[0]) )
Pt = tt.kron(Pt, tt.tensor( gamma_pdf(pts2,alpha_prior[1],beta_prior[1]) ) )
Pt = tt.kron(Pt, tt.tensor( gamma_pdf(pts3,alpha_prior[2],beta_prior[2]) ) )
Pt = tt.kron(Pt, tt.tensor( gamma_pdf(pts4,alpha_prior[3],beta_prior[3]) ) )
Pt = tt.kron(Pt, tt.tensor( gamma_pdf(pts5,alpha_prior[4],beta_prior[4]) ) )

# Pt = tt.tensor(np.ones([Nl,Nl]))
WS = tt.kron(tt.kron(tt.kron(tt.tensor(ws1),tt.tensor(ws2)),tt.kron(tt.tensor(ws3),tt.tensor(ws4))) , tt.tensor(ws5) )
Z = tt.sum(Pt*WS)
Pt = Pt * (1/Z)
Pt_prior = Pt 
P = tt.kron(tt.tensor(x0),Pt)
P = P * (1/tt.sum(P*tt.kron(tt.ones(N),WS)))
plt.figure()
plt.plot(pts1,gamma_pdf(pts1,alpha_prior[0],beta_prior[0]))
plt.figure()
plt.plot(pts2,gamma_pdf(pts2,alpha_prior[1],beta_prior[1]))
plt.figure()
plt.plot(pts3,gamma_pdf(pts3,alpha_prior[2],beta_prior[2]))
plt.figure()
plt.plot(pts4,gamma_pdf(pts4,alpha_prior[3],beta_prior[3]))
plt.figure()
plt.plot(pts5,gamma_pdf(pts5,alpha_prior[4],beta_prior[4]))
plt.pause(0.05)


#%% integrator 

if qtt:
    A_qtt = ttm2qttm(A_tt)
    fwd_int = ttInt(A_qtt, epsilon = 1e-6, N_max = 8, dt_max = 1.0,method='cheby')
    ws_qtt = tt2qtt(WS)
    P = tt2qtt(P)
else:
    fwd_int = ttInt(A_tt, epsilon = 1e-6, N_max = 64, dt_max = 1.0,method='crank–nicolson')
    
Pts = []
Pjs_fwd = []
print('Starting...')
tme_total = datetime.datetime.now()
for i in range(1,No):
    
    y = observations_noise[i,:]

    
    PO = tt.tensor(np.exp(-0.5*(y[0]-np.arange(N[0]))**2/sigma**2))
    PO = tt.kron(PO, tt.tensor(np.exp(-0.5*(y[1]-np.arange(N[1]))**2/sigma**2)))
    PO = tt.kron(PO, tt.tensor(np.exp(-0.5*(y[2]-np.arange(N[2]))**2/sigma**2)))
    PO = tt.kron(PO, tt.tensor(np.exp(-0.5*(y[3]-np.arange(N[3]))**2/sigma**2)))

    PO = PO * (1/tt.sum(PO))
    PO = tt.kron(PO,tt.ones([Nl]*5))
    
    if qtt: PO = tt2qtt(PO)
    
    print('new observation ',i,' at time ',time_observation[i],' ',y)
    
    tme = datetime.datetime.now()
    P = fwd_int.solve(P, dT, intervals = Nbs,qtt = qtt)
    tme = datetime.datetime.now() - tme
    
    print('\tmax rank ',max(P.r))
    Ppred = P
    Ppost = PO * Ppred
    Ppost = Ppost.round(1e-10,500)
    print('\tmax rank (after observation) ',max(Ppost.r))
    
    if not qtt:
        Ppost = Ppost * (1/tt.sum(Ppost * tt.kron(tt.ones(N),WS)))
        Pt = tt.sum(tt.sum(tt.sum(tt.sum(Ppost,0),0),0),0) 
        Z = tt.sum(Pt*WS)
        Pt = Pt * (1/Z)
        Pt = Pt.round(1e-10,200)
    
    else:
        Ppost = Ppost * (1/tt.sum(Ppost * tt.kron(tt.ones(int(np.sum(np.log2(N)))*[2]),ws_qtt)))
        Pt = Ppost
        for i in range(int(np.sum(np.log2(N)))): Pt = tt.sum(Pt,0) 
        Z = tt.sum(Pt*ws_qtt)
        Pt = Pt * (1/Z)
        Pt = Pt.round(1e-10,500)
    
#    print(Pt.r)
    Pts.append(Pt.copy())
    
    if qtt: Pt = qtt2tt(Pt,[Nl]*5)
    
    E1 = tt.sum(Pt * tt.kron(tt.tensor(pts1),tt.ones([Nl]*4)) * WS)
    E2 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]),tt.tensor(pts2)),tt.ones([Nl]*3)) * WS)
    E3 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*2),tt.tensor(pts3)),tt.ones([Nl]*2)) * WS)
    E4 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*3),tt.tensor(pts4)),tt.ones([Nl])) * WS)
    E5 = tt.sum(Pt * tt.kron(tt.ones([Nl]*4),tt.tensor(pts5)) * WS)
    E = np.array([E1,E2,E3,E4,E5])
    
    V1 = tt.sum(Pt * tt.kron(tt.tensor(pts1**2),tt.ones([Nl]*4)) * WS) - E1**2
    V2 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]),tt.tensor(pts2**2)),tt.ones([Nl]*3)) * WS) - E2**2
    V3 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*2),tt.tensor(pts3**2)),tt.ones([Nl]*2)) * WS) - E3**2
    V4 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*3),tt.tensor(pts4**2)),tt.ones([Nl])) * WS) - E4**2
    V5 = tt.sum(Pt * tt.kron(tt.ones([Nl]*4),tt.tensor(pts5**2)) * WS) - E5**2
    V = np.array([V1,V2,V3,V4,V5])

    print('\tExpected value computed posterior ' ,E)
    print('\tVariance computed posterior       ' ,V)

    P = Ppost
    print('\tposterior size ',sum([elem.size for elem in P.to_list(P)])*8 / 1000000,' MB')
    print('\telapsed ',tme)
    
    # P12 = 

    
    # plt.pause(0.05)

Pt_fwd = Pt

tme_total = datetime.datetime.now() - tme_total

print()
print('Total time ',tme_total)

#%% show 
Post = Pt_fwd
Prior = Pt_prior

Pt = Pt_fwd
E1 = tt.sum(Pt * tt.kron(tt.tensor(pts1),tt.ones([Nl]*4)) * WS)
E2 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]),tt.tensor(pts2)),tt.ones([Nl]*3)) * WS)
E3 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*2),tt.tensor(pts3)),tt.ones([Nl]*2)) * WS)
E4 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*3),tt.tensor(pts4)),tt.ones([Nl])) * WS)
E5 = tt.sum(Pt * tt.kron(tt.ones([Nl]*4),tt.tensor(pts5)) * WS)
E = np.array([E1,E2,E3,E4,E5])

V1 = tt.sum(Pt * tt.kron(tt.tensor(pts1**2),tt.ones([Nl]*4)) * WS) - E1**2
V2 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]),tt.tensor(pts2**2)),tt.ones([Nl]*3)) * WS) - E2**2
V3 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*2),tt.tensor(pts3**2)),tt.ones([Nl]*2)) * WS) - E3**2
V4 = tt.sum(Pt * tt.kron(tt.kron(tt.ones([Nl]*3),tt.tensor(pts4**2)),tt.ones([Nl])) * WS) - E4**2
V5 = tt.sum(Pt * tt.kron(tt.ones([Nl]*4),tt.tensor(pts5**2)) * WS) - E5**2
V = np.array([V1,V2,V3,V4,V5])

import pyswarm

def goal_function(thetuta):

    L1 = np.array([ lagrange(thetuta[0],i,pts1) for i in range(pts1.size) ] )
    L2 = np.array([ lagrange(thetuta[1],i,pts2) for i in range(pts2.size) ] )
    L3 = np.array([ lagrange(thetuta[2],i,pts3) for i in range(pts3.size) ] )
    L4 = np.array([ lagrange(thetuta[3],i,pts4) for i in range(pts4.size) ] )
    L5 = np.array([ lagrange(thetuta[4],i,pts5) for i in range(pts5.size) ] )

    val = tt.dot(Post,tt.mkron(tt.tensor(L1.flatten()), tt.tensor(L2.flatten()), tt.tensor(L3.flatten()), tt.tensor(L4.flatten()), tt.tensor(L5.flatten())))
    return -val

theta_mode, _ = pyswarm.pso(goal_function, np.array(param_range)[:,0], np.array(param_range)[:,1])

print('Exact rates:                      ',rates)
print('')
print('Expected value computed posterior ' ,E)
print('Variance computed posterior       ' ,V)
print('Computed modes:                   ',theta_mode)
print('')
print('Expected value prior              ' ,alpha_prior/beta_prior)
print('Variance computed prior           ' ,alpha_prior/beta_prior/beta_prior)
print('')


L1 = np.array([ lagrange(np.linspace(param_range[0][0],param_range[0][1],256),i,pts1) for i in range(pts1.size) ] )
L2 = np.array([ lagrange(np.linspace(param_range[1][0],param_range[1][1],256),i,pts2) for i in range(pts2.size) ] )
L3 = np.array([ lagrange(np.linspace(param_range[2][0],param_range[2][1],256),i,pts3) for i in range(pts3.size) ] )
L4 = np.array([ lagrange(np.linspace(param_range[3][0],param_range[3][1],256),i,pts4) for i in range(pts4.size) ] )
L5 = np.array([ lagrange(np.linspace(param_range[4][0],param_range[4][1],256),i,pts5) for i in range(pts5.size) ] )


# theta1, theta2 = np.meshgrid(np.linspace(rates[0]*lb,rates[0]*ub,128),np.linspace(rates[3]*lb,rates[3]*ub,128))

def mode_prod(t,Ms):
    cores = t.to_list(t)
    for i in range(len(cores)):
        cores[i] = np.einsum('ijk,jl->ilk',cores[i],Ms[i])
    return tt.tensor().from_list(cores)    
    
Post1 = mode_prod(Post,[L1,ws2.reshape([-1,1]),ws3.reshape([-1,1]),ws4.reshape([-1,1]),ws5.reshape([-1,1])]).full().flatten()
Post2 = mode_prod(Post,[ws1.reshape([-1,1]),L2,ws3.reshape([-1,1]),ws4.reshape([-1,1]),ws5.reshape([-1,1])]).full().flatten()
Post3 = mode_prod(Post,[ws1.reshape([-1,1]),ws2.reshape([-1,1]),L3,ws4.reshape([-1,1]),ws5.reshape([-1,1])]).full().flatten()
Post4 = mode_prod(Post,[ws1.reshape([-1,1]),ws2.reshape([-1,1]),ws3.reshape([-1,1]),L4,ws5.reshape([-1,1])]).full().flatten()
Post5 = mode_prod(Post,[ws1.reshape([-1,1]),ws2.reshape([-1,1]),ws3.reshape([-1,1]),ws4.reshape([-1,1]),L5]).full().flatten()


Prior1 = mode_prod(Prior,[L1,ws2.reshape([-1,1]),ws3.reshape([-1,1]),ws4.reshape([-1,1]),ws5.reshape([-1,1])]).full().flatten()
Prior2 = mode_prod(Prior,[ws1.reshape([-1,1]),L2,ws3.reshape([-1,1]),ws4.reshape([-1,1]),ws5.reshape([-1,1])]).full().flatten()
Prior3 = mode_prod(Prior,[ws1.reshape([-1,1]),ws2.reshape([-1,1]),L3,ws4.reshape([-1,1]),ws5.reshape([-1,1])]).full().flatten()
Prior4 = mode_prod(Prior,[ws1.reshape([-1,1]),ws2.reshape([-1,1]),ws3.reshape([-1,1]),L4,ws5.reshape([-1,1])]).full().flatten()
Prior5 = mode_prod(Prior,[ws1.reshape([-1,1]),ws2.reshape([-1,1]),ws3.reshape([-1,1]),ws4.reshape([-1,1]),L5]).full().flatten()




plt.figure()
plt.plot(np.linspace(param_range[0][0],param_range[0][1],256),Post1)
plt.plot(np.linspace(param_range[0][0],param_range[0][1],256),Prior1,'g--')
plt.axvline(rates[0],c='r',linestyle=':')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'probability density')
plt.legend(['Posterior','Prior','True parameter'])

plt.figure()
plt.plot(np.linspace(param_range[1][0],param_range[1][1],256),Post2)
plt.plot(np.linspace(param_range[1][0],param_range[1][1],256),Prior2,'g--')
plt.scatter(pts2,pts2*0)
plt.axvline(rates[1],c='r',linestyle=':')
plt.xlabel(r'$\theta_2$')
plt.ylabel(r'probability density')
plt.legend(['Posterior','Prior','True parameter'])

plt.figure()
plt.plot(np.linspace(param_range[2][0],param_range[2][1],256),Post3)
plt.plot(np.linspace(param_range[2][0],param_range[2][1],256),Prior3,'g--')
plt.axvline(rates[2],c='r',linestyle=':')
plt.xlabel(r'$\theta_3$')
plt.ylabel(r'probability density')
plt.legend(['Posterior','Prior','True parameter'])

plt.figure()
plt.plot(np.linspace(param_range[3][0],param_range[3][1],256),Post4)
plt.plot(np.linspace(param_range[3][0],param_range[3][1],256),Prior4,'g--')
plt.axvline(rates[3],c='r',linestyle=':')
plt.xlabel(r'$\theta_4$')   
plt.ylabel(r'probability density')
plt.legend(['Posterior','Prior','True parameter'])

plt.figure()
plt.plot(np.linspace(param_range[4][0],param_range[4][1],256),Post5)
plt.plot(np.linspace(param_range[4][0],param_range[4][1],256),Prior5,'g--')
plt.axvline(rates[4],c='r',linestyle=':')
plt.xlabel(r'$\theta_4$')
plt.ylabel(r'probability density')
plt.legend(['Posterior','Prior','True parameter'])



