#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:56:29 2021

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
# speices S,E,I,Q,R

AV = 800
rates = np.array([0.04,0.4,0.4,0.04,0.12,0.8765,0.01,0.01,0.1])
# 0.1,0.5,1.0,0.01,0.01,0.1,0.4
Pre =[[1,0,1,0,0]]    # S+I->E+I
Pre.append( [0,1,0,0,0])    # E->I
Pre.append( [0,0,1,0,0])    # I->Q
Pre.append( [0,0,0,1,0])     # Q->none
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

         

# Props = [ lambda x: x[:,5]*x[:,3], lambda x: x[:,4], lambda x: x[:,3], lambda x: x[:,3],  lambda x: x[:,2],  lambda x: x[:,1], lambda x: x[:,3], lambda x: x[:,2], lambda x: x[:,1], lambda x: x[:,3], lambda x: x[:,2], lambda x: x[:,1]]
# Pre=Pre[:,::-1]
# Post=Post[:,::-1]


# construct the model and the CME operator
N = [128]*5 # state truncation
N = [128,64,64,32,32] # state truncation
# N = N[::-1]
mdl = CME(N, Pre,Post,rates,Props)
s0 = [90,0,4,0,0]
# s0 = s0[::-1]
Tend = 4
qtt = True
Att = mdl.construct_generator_tt(as_list = False)

reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(s0),Tend,Pre,Post-Pre,rates)

plt.figure()
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,0],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,1],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,2],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,3],2)[:-1])
plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,4],2)[:-1])
plt.legend(['S','E','I','Q','R'])
plt.pause(0.05)

P = SingularPMF(N,s0)

if qtt:
    A_qtt = ttm2qttm(Att).round(1e-12)
    integrator = ttInt(A_qtt, epsilon = 1e-5, N_max = 8, dt_max = 1.0,method='cheby')
    P = tt2qtt(P)
else:
    integrator = ttInt(Att, epsilon = 1e-7, N_max = 64, dt_max = 1.0,method='crank–nicolson')
   
time_tt = datetime.datetime.now()
for i in range(32):

    dt = 0.5/4
    tme = datetime.datetime.now()
    P= integrator.solve(P, dt, intervals = 1,qtt=True, verb=True, rounding = False)
    tme = datetime.datetime.now() - tme
    # P = P.round(1e-10)
    print(i,' time ',tme,' size [MB] ',tt_size(P)*8/1e6,' rank ',P.r)
    # P = P.round(1e-8,80)
time_tt = datetime.datetime.now() - time_tt
print('TT time ',time_tt)

P = qtt2tt(P,N)



