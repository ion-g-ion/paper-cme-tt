#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:53:03 2020

@author: ion
"""





import tensorflow as tf
import t3f
import numpy as np
import matplotlib.pyplot as plt
from CME import CME, Gillespie, Observations_grid
import timeit
import scipy.integrate
import numba
import scipy.sparse
import tt_extra as tte
import tt
from ttInt import ttInt
import tt.amen
from tt_aux import *
import datetime


# define reaction 
rates = np.array([0.1,0.5,1.0,0.01,0.01,0.01,0.4])
Pre =np.array( [[1,0,1,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])
Post = np.array([[0,1,1,0],[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1],[1,0,0,0]])
Props = [ lambda x: x[:,0]*x[:,2] , lambda x: x[:,1]  , lambda x: x[:,2]  , lambda x: x[:,0] , lambda x: x[:,1] , lambda x: x[:,2] , lambda x: x[:,0]*0+1 ]

# construct the model and the CME operator
N = [128]*4 # state truncation
Initial = [50,5,0,0]

x0 = np.zeros(N)
x0[tuple(Initial)] = 1.0


qtt = True

mdl = CME(N, Pre,Post,rates,Props)
A_tt = mdl.construct_generator_tt()



#%% Time grid 

No = 33
dT = 0.3125

time_observation = np.arange(No)*dT

np.random.seed(123456)


# states_fine = mdl.ssa(np.array(Initial),fine_time_grid,1)
# states = np.array( [states_fine[np.where(fine_time_grid == t)[0][0],:,0] for t in time_observation] )

reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(Initial),time_observation[-1],Pre,Post-Pre,rates)
states = Observations_grid(time_observation, reaction_time, reaction_jumps)


#%% Observation
s1 = 0.1
s2 = 0.1
s3 = 0.1
s4 = 0.05
observations = states + np.hstack((np.random.normal(0,1,(states.shape[0],3)),np.random.normal(0,0.000001,[states.shape[0],1])))
observations = np.hstack((np.random.lognormal(np.log(states[:,0]+1),s1).reshape([-1,1]) , np.random.lognormal(np.log(states[:,1]+1),s2).reshape([-1,1]) , np.random.lognormal(np.log(states[:,2]+1),s3).reshape([-1,1]) , np.random.lognormal(np.log(states[:,3]+1),s4).reshape([-1,1])))
noise_model = lambda x,y,s : 1/(y*s*np.sqrt(2*np.pi)) * np.exp(-(np.log(y)-np.log(x+1))**2/(2*s**2))

#%% Forward pass

tme_total = datetime.datetime.now()
P = tt.tensor(x0)


if qtt:
    A_qtt = ttm2qttm(A_tt)
    fwd_int = ttInt(A_qtt, epsilon = 1e-6, N_max = 8, dt_max = 1.0,method='cheby')
    P = tt2qtt(P)
else:
    fwd_int = ttInt(A_tt, epsilon = 1e-6, N_max = 8, dt_max = 1.0,method='cheby')

P_fwd = [P.copy()]
ranks_fwd = [[0,max(P.r)]]
time = 0
for i in range(1,No):
    
    
    y = observations[i,:]
        
    Po1 = noise_model(np.arange(N[0]),y[0],s1)
    Po2 = noise_model(np.arange(N[1]),y[1],s2)
    Po3 = noise_model(np.arange(N[2]),y[2],s3)
    Po4 = noise_model(np.arange(N[3]),y[3],s4)
    
    Po = tt.kron(tt.kron(tt.tensor(Po1),tt.tensor(Po2)),tt.kron(tt.tensor(Po3),tt.tensor(Po4)))
    Po = Po * (1/tt.sum(Po))
    
    if qtt: Po = tt2qtt(Po)
    
    tme = timeit.time.time()
    P = fwd_int.solve(P, dT, intervals = 6,return_all=True,qtt=qtt) 
    # for p in P:
    #     print('\t',p.r)
    
    if qtt :
        P_fwd += [p[tuple([slice(None,None,None)]*len(A_qtt.n)+[-1])].round(1e-8) for p in P] 
        for k in range(len(P)):
            time+=dT/len(P)
            ranks_fwd += [[time,max(P[k][tuple([slice(None,None,None)]*len(A_qtt.n)+[-1])].round(1e-10).r)]]
        P = P[-1][tuple([slice(None,None,None)]*len(A_qtt.n)+[-1])]
    else:
        P_fwd += [p[:,:,:,:,-1] for p in P] 
        P = P[-1][:,:,:,:,-1]
        
    tme = timeit.time.time() - tme
    print('\tmax rank before observation ',max(P.round(1e-10).r),' not rounded ',max(P.r))
    P_next = P * Po
    P_next = P_next.round(1e-10)
    P_next = P_next * (1/tt.sum(P_next))
    print('\tMax rank after observation ',max(P_next.r))
    ranks_fwd += [[time,max(P_next.r)]]
    P = P_next 
    
    # P_fwd.append(P)
    
    print('observation ',i+1,' at time ',time_observation[i],' is ' ,y,' time elapsed',tme,' s')

ranks_fwd = np.array(ranks_fwd)
    
#%% Backward pass

ranks_bck = []
    
P = tt2qtt(tt.ones(N)*(1/np.prod(N))) if qtt else tt.ones(N)*(1/np.prod(N))
P_bck = []

if qtt:
    A_qtt = ttm2qttm(A_tt.T)
    bck_int = ttInt(A_qtt, epsilon = 1e-6, N_max = 8, dt_max = 1.0,method='cheby')
else:
    bck_int = ttInt(A_tt.T, epsilon = 1e-6, N_max = 8, dt_max = 1.0,method='cheby')



ranks_bck = []
for i in range(No-1,0,-1):
    
    y = observations[i,:]
      
    tme = timeit.time.time() 

    Po1 = noise_model(np.arange(N[0]),y[0],s1)
    Po2 = noise_model(np.arange(N[1]),y[1],s2)
    Po3 = noise_model(np.arange(N[2]),y[2],s3)
    Po4 = noise_model(np.arange(N[3]),y[3],s4)
    Po = tt.kron(tt.kron(tt.tensor(Po1),tt.tensor(Po2)),tt.kron(tt.tensor(Po3),tt.tensor(Po4)))
    Po = Po * (1/tt.sum(Po))
    
    if qtt: Po = tt2qtt(Po)
    
    print('\tmax rank before observation ',max(P.r),' rank of observation ',max(Po.r))
    P = (P * Po).round(1e-10)
    print('\tmax rank after observation ',max(P.r))
    P = bck_int.solve(P, dT, intervals = 6,return_all=True,qtt = qtt)
    
    if qtt :
        P_bck += [p[tuple([slice(None,None,None)]*len(A_qtt.n)+[0])].round(1e-8) for p in P] 
        P = P[-1][tuple([slice(None,None,None)]*len(A_qtt.n)+[-1])]
    else:
        P_bck += [p[:,:,:,:,0] for p in P]
        P = P[-1][:,:,:,:,-1]
        
    tme = timeit.time.time() - tme
    
    P_next = P
    # P_next = P * Po
    P_next = P_next.round(1e-10)
    P_next = P_next * (1/tt.sum(P_next))
    
    P = P_next 
    
    # P_bck.append(P)
    
    print('observation ',i+1,' at time ',time_observation[i],' is ' ,y,' time elapsed',tme,' s')

P_bck.append(P)

P_bck = P_bck[::-1]

#%% Combine messages

P_hmm = []

x1 = tt.kron( tt.kron(tt.tensor(np.arange(N[0])),tt.tensor(np.ones(N[1]))), tt.kron(tt.tensor(np.ones(N[2])),tt.tensor(np.ones(N[3]))))
x2 = tt.kron( tt.kron(tt.tensor(np.ones(N[0])),tt.tensor(np.arange(N[1]))), tt.kron(tt.tensor(np.ones(N[2])),tt.tensor(np.ones(N[3]))))
x3 = tt.kron( tt.kron(tt.tensor(np.ones(N[0])),tt.tensor(np.ones(N[1]))), tt.kron(tt.tensor(np.arange(N[2])),tt.tensor(np.ones(N[3]))))
x4 = tt.kron( tt.kron(tt.tensor(np.ones(N[0])),tt.tensor(np.ones(N[1]))), tt.kron(tt.tensor(np.ones(N[2])),tt.tensor(np.arange(N[3]))))

Es = []
Vs = []

if qtt:
    x1 = tt2qtt(x1)
    x2 = tt2qtt(x2)
    x3 = tt2qtt(x3)
    x4 = tt2qtt(x4)
    
for i in range(len(P_bck)):
    print(i)
    Pf = P_fwd[i]
    Pb = P_bck[i]
    # if qtt:
    #     Pf = qtt2tt(Pf,N)
    #     Pb = qtt2tt(Pb,N)
        
    Z = tt.dot(Pf,Pb)
    
    mean = [tt.dot(Pf,Pb*x1)/Z, tt.dot(Pf,Pb*x2)/Z, tt.dot(Pf,Pb*x3)/Z, tt.dot(Pf,Pb*x4)/Z]
    var = [tt.dot(Pf*x1,Pb*x1)/Z-mean[0]**2, tt.dot(Pf*x2,Pb*x2)/Z-mean[1]**2, tt.dot(Pf*x3,Pb*x3)/Z-mean[2]**2, tt.dot(Pf*x4,Pb*x4)/Z-mean[3]**2]

    Es.append(mean)
    Vs.append(var)
    
#    P_hmm.append(P)
    
Es = np.array(Es)
Vs = np.sqrt(np.array(Vs))
Vs = np.nan_to_num(Vs)
tme_total = datetime.datetime.now() - tme_total



print()
print('Elapsed time',tme_total)
print('Total size forward messages ',sum([tt_size(p) for p in P_fwd])*8/1e6,' MB')
print('Total size backward messagea ',sum([tt_size(p) for p in P_bck])*8/1e6,' MB')
print()

#%% Plots
import tikzplotlib
n  = Es.shape[0]
time_plot = np.linspace(0,(No-1)*dT,n)
plt.figure()
plt.fill(np.concatenate((time_plot,time_plot[::-1])), np.concatenate((Es[:,0] - Vs[:,0],Es[::-1,0] + Vs[::-1,0])), 'grey') 
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,0],2)[:-1],'b',linewidth=1,label='true state')
plt.plot(time_plot,Es[:,0],'r--',linewidth=1,label='mean')
plt.scatter(time_observation,observations[:,0],s=5,c='k',label='observations')
plt.xlabel(r'$t$')
plt.ylabel(r'#individuals')
plt.legend(['true state','mean','std','observations'])
# tikzplotlib.save('./plots/seir_filter_S.tex')
# plt.errorbar(time_observation, Es[:,0], yerr=Vs[:,0],c='r')
tikzplotlib.save('./plots/seir_filter_S.tex',table_row_sep=r"\\")

plt.figure()
plt.fill(np.concatenate((time_plot,time_plot[::-1])), np.concatenate((Es[:,1] - Vs[:,1],Es[::-1,1] + Vs[::-1,1])), 'grey') 
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,1],2)[:-1],'b',linewidth=1)
plt.plot(time_plot,Es[:,1],'r--',linewidth=1)
plt.scatter(time_observation,observations[:,1],s=5,c='k')
plt.legend(['true state','mean','std','observations'])
plt.xlabel(r'$t$')
plt.ylabel(r'#individuals')
# tikzplotlib.save('./plots/seir_filter_E.tex')
# plt.errorbar(time_observation, Es[:,0], yerr=Vs[:,0],c='r')
tikzplotlib.save('./plots/seir_filter_E.tex',table_row_sep=r"\\")

plt.figure()
plt.fill(np.concatenate((time_plot,time_plot[::-1])), np.concatenate((Es[:,2] - Vs[:,2],Es[::-1,2] + Vs[::-1,2])), 'grey') 
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,2],2)[:-1],'b',linewidth=1)
plt.plot(time_plot,Es[:,2],'r--',linewidth=1)
plt.scatter(time_observation,observations[:,2],s=5,c='k',marker='x')
plt.legend(['true state','mean','std','observations'])
plt.xlabel(r'$t$')
plt.ylabel(r'#individuals')
# tikzplotlib.save('./plots/seir_filter_I.tex')
# plt.errorbar(time_observation, Es[:,0], yerr=Vs[:,0],c='r')
tikzplotlib.save('./plots/seir_filter_I.tex',table_row_sep=r"\\")

plt.figure()
plt.fill(np.concatenate((time_plot,time_plot[::-1])), np.concatenate((Es[:,3] - Vs[:,3],Es[::-1,3] + Vs[::-1,3])), 'grey') 
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,3],2)[:-1],'b',linewidth=1)
plt.plot(time_plot,Es[:,3],'r--',linewidth=1)
plt.scatter(time_observation,observations[:,3],marker="x", c="k",  s=5)
plt.legend(['true state','mean','std','observations'])
plt.xlabel(r'$t$')
plt.ylabel(r'#individuals')
tikzplotlib.save('./plots/seir_filter_R.tex',table_row_sep=r"\\")
# plt.errorbar(time_observation, Es[:,0], yerr=Vs[:,0],c='r')

plt.figure()
plt.plot(ranks_fwd[:,0],ranks_fwd[:,1])
plt.xlabel(r'$t$')
plt.ylabel(r'maximum TT-rank')
plt.grid()
tikzplotlib.save('./plots/seir_filter_ranks.tex',table_row_sep=r"\\")

size_tt = lambda ttt : sum([c.size for c in ttt.to_list(ttt)])