#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:50:22 2020

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


# speices RNA,M,DNA.D,DNA,D,DNA.2D

AV = 6.022145 * 1e23  * 1e-15
rates = np.array([0.043,0.0007,0.0715,0.0039,0.012*1e9 / AV,0.4791,0.00012*1e9/AV,0.8765*1e-11,0.05*1e9/AV,0.5])

Pre =np.array( [[1,0,0,0,0,0], \
                [0,1,0,0,0,0], \
                [0,0,1,0,0,0], \
                [1,0,0,0,0,0], \
                [0,0,0,1,1,0], \
                [0,0,1,0,0,0], \
                [0,0,1,0,1,0],\
                [0,0,0,0,0,1],\
                [0,2,0,0,0,0],\
                [0,0,0,0,1,0]])
Post = np.array([[1,1,0,0,0,0],\
                 [0,0,0,0,0,0],\
                 [1,0,1,0,0,0],\
                 [0,0,0,0,0,0],\
                 [0,0,1,0,0,0],\
                 [0,0,0,1,1,0],\
                 [0,0,0,0,0,1],\
                 [0,0,1,0,1,0],\
                 [0,0,0,0,1,0],\
                 [0,2,0,0,0,0]])
    
Props = [ lambda x: x[:,0],\
         lambda x: x[:,1],\
         lambda x: x[:,2],\
         lambda x: x[:,0],\
         lambda x: x[:,3]*x[:,4],\
         lambda x: x[:,2],\
         lambda x: x[:,2]*x[:,4],\
         lambda x: x[:,5],\
         lambda x: 0.5*x[:,1]*(x[:,1]-1),\
         lambda x: x[:,4]]




# construct the model and the CME operator
N = [16,32,4,4,64,4] # state truncation
# N = [64]*6 # state truncation
mdl = CME(N, Pre,Post,rates,Props)
s0 = [0,2,0,2,6,0]
Tend = 1000
qtt = True
Att = mdl.construct_generator_tt(as_list = False)

reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(s0),Tend,Pre,Post-Pre,rates)

plt.figure()
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,0],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,1],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,2],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,3],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,4],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,5],2)[:-1])

# import sys
# sys.exit()

evector = lambda n,i : np.eye(n)[:,i]

P = tt.kron(tt.tensor(evector(N[0],s0[0])),tt.tensor(evector(N[1],s0[1])))
P = tt.kron(P,tt.tensor(evector(N[2],s0[2])))
P = tt.kron(P,tt.tensor(evector(N[3],s0[3])))
P = tt.kron(P,tt.tensor(evector(N[4],s0[4])))
P = tt.kron(P,tt.tensor(evector(N[5],s0[5])))

if qtt:
    A_qtt = ttm2qttm(Att)
    integrator = ttInt(A_qtt, epsilon = 1e-7, N_max = 8, dt_max = 1.0,method='cheby')
    P = tt2qtt(P)
else:
    integrator = ttInt(Att, epsilon = 1e-7, N_max = 64, dt_max = 1.0,method='crank–nicolson')
   

for i in range(25):

    dt = 12
    tme = timeit.time.time() 
    P= integrator.solve(P, dt, intervals = 12,qtt=True)
    tme = timeit.time.time() - tme
    print(i,' time ',tme,'  ',P.r)
    # P = P.round(1e-8,80)
    
P = qtt2tt(P,N)
P_D = tt.sum(tt.sum(tt.sum(tt.sum(tt.sum(P,0),0),0),0),1).full()

plt.figure()
plt.plot(P_D)

    