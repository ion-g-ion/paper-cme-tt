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


rates = np.array([0.1,0.5,1.0,0.01,0.01,0.01,0.4])
Pre =np.array( [[1,0,1,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])
Post = np.array([[0,1,1,0],[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1],[1,0,0,0]])
Props = [ lambda x: x[:,0]*x[:,2] , lambda x: x[:,1]  , lambda x: x[:,2]  , lambda x: x[:,0] , lambda x: x[:,1] , lambda x: x[:,2] , lambda x: x[:,0]*0+1 ]


         

# construct the model and the CME operator
N = [128,64,64,64] # state truncation

Tend = 8
Nt = 4

mdl = CME(N, Pre,Post,rates,Props)
s0 = [50,4,0,0]
qtt = True
Att = mdl.construct_generator_tt(as_list = False)

reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(s0),Tend,Pre,Post-Pre,rates)

plt.figure()
for i in range(4): plt.plot(np.repeat(reaction_time,2)[1:],np.repeat(reaction_jumps[:,i],2)[:-1])
plt.legend(['S','E','I','R'])
plt.pause(0.05)

P = SingularPMF(N,s0)

if qtt:
    A_qtt = ttm2qttm(Att).round(1e-12)
    integrator = ttInt(A_qtt, epsilon = 1e-5, N_max = 8, dt_max = 1.0,method='cheby')
    P = tt2qtt(P)
else:
    integrator = ttInt(Att, epsilon = 1e-7, N_max = 64, dt_max = 1.0,method='crank–nicolson')
   
time_tt = datetime.datetime.now()
time = 0
Ps = [P.copy()]
P0 = P.copy()
for i in range(Nt):

    dt = Tend/Nt
    tme = datetime.datetime.now()
    P= integrator.solve(P, dt, intervals = 8,qtt=True, verb=True, rounding = False)
    tme = datetime.datetime.now() - tme
    P = P.round(1e-10)
    print(i,' time ',tme,' size [MB] ',tt_size(P)*8/1e6,' rank ',P.r)
    time += dt
    Ps.append(P)
    
time_tt = datetime.datetime.now() - time_tt
print('TT time ',time_tt)

P = qtt2tt(P,N)
Pend = P.full()

import sys
sys.exit()
#%% reference
print('Reference...')
tme_ode45 = timeit.time.time()
mdl = CME([80,64,64,64], Pre,Post,rates,Props)
P0 = SingularPMF([80,64,64,64],s0)
mdl.construct_generator2(to_tf=False)
Gen = mdl.gen
def func(t,y):
    print(t)
    return Gen.dot(y)

# solve CME

res = scipy.integrate.solve_ivp(func,[0,Tend],P0.full().flatten(),t_eval=[0,Tend],max_step=dt/500)
Pt = res.y.reshape([80,64,64,64]+[-1])
tme_ode45 = timeit.time.time() - tme_ode45

P_ref = Pt[:,:,:,:,-1]



#%% errors
residual = (Pend[:80,:,:,:]-P_ref)
# residual = residual[:40,:40,:40,:40]
print('Mean rel error ',np.mean(np.abs(residual))/np.max(np.abs(Pend)))
print('Max rel error ',np.max(np.abs(residual))/np.max(np.abs(Pend)))

# P_ref[64:,:,:,:] = 0
# P_ref[:,64:,:,:] = 0

Pend = Pend[:80,:64,:64,:64]

#%% plots
import tikzplotlib
plt.figure()
plt.imshow(Pend.sum(2).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('./plots/SE_marginal.tex')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(2).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('./plots/SE_marginal_err.tex')

plt.figure()
plt.imshow(Pend.sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal.tex')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal_err.tex')

plt.figure()
plt.imshow(Pend.sum(0).sum(0).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_3$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/IR_marginal.tex')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(0).sum(0).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_3$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/IR_marginal_err.tex')

plt.figure()
plt.imshow(Pend.sum(1).sum(1).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/SR_marginal.tex')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(1).sum(1).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/SR_marginal_err.tex')

plt.figure()
plt.imshow(qtt2tt(Ps[0],N).full().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal_0.tex')

plt.figure()
plt.imshow(qtt2tt(Ps[1],N).full().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal_2.tex')

plt.figure()
plt.imshow(qtt2tt(Ps[2],N).full().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal_4.tex')

plt.figure()
plt.imshow(qtt2tt(Ps[3],N).full().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal_6.tex')