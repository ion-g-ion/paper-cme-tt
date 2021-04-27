#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:01:49 2020

@author: ion
"""

import tensorflow as tf
import t3f
import numpy as np
import matplotlib.pyplot as plt
from CME import CME
import timeit
import scipy.integrate
import numba
import scipy.sparse
from tt_extra import mat_to_tt
import tt
import tt.amen
from ttInt import ttInt

# define reaction 
rates = np.array([0.015,0.002,0.1,0.01])
Pre =np.array( [[1,0],[1,0],[0,0],[0,1]])
Post = np.array([[1,1],[0,0],[1,0],[0,0]])
Props = [ lambda x: x[:,0], lambda x: x[:,0]  , lambda x: x[:,0]*0+1  , lambda x: x[:,1] ]



# construct the model and the CME operator
N = [80,120] # state truncation
mdl = CME(N, Pre,Post,rates,Props)

mdl.construct_generator2(to_tf=False)

A_tt = mdl.construct_generator_tt()

Initial = [2,4]
P0 = np.zeros(N)
P0[Initial[0],Initial[1]] = 1.0
P0_tt = tt.tensor(P0)

dT = 128
Nt = 8
time = np.arange(Nt+1) * dT

# Reference solution
print('Reference solution...')
tme_ode45 = timeit.time.time()
mdl.construct_generator2(to_tf=False)
Gen = mdl.gen
def func(t,y):
    return Gen.dot(y)

# solve CME
res = scipy.integrate.solve_ivp(func,[0,time[-1]],P0.flatten(),t_eval=time,max_step=dT/10000)
Pt = res.y.reshape(N+[-1])
P_ref = Pt[:,:,-1]
tme_ode45 = timeit.time.time() - tme_ode45


# convergence test
print('Implicit Euler...')
err_implicit = []
refinements_implicit = [16,32,64,128,256,512]
for nt in refinements_implicit:
    
    fwd_int = ttInt(A_tt, epsilon = 1e-9, N_max = nt, dt_max = 100.0)

    P_tt = P0_tt
    for i in range(Nt):
        P_tt = fwd_int.solve(P_tt, dT, intervals = 1)

    P = P_tt.full().reshape(N)
    
    err = np.max(np.abs(P-P_ref)) / np.max(np.abs(P_ref))
    err_implicit.append(err)
    print('nt ',nt,' error inf ',err)
    
    
# convergence test
print('Crank Nicolson...')
err_cn = []
refinements_cn = [16,32,64,128,256,512]
for nt in refinements_cn:
    
    fwd_int = ttInt(A_tt, epsilon = 1e-11, N_max = nt, dt_max = 100.0,method='crank–nicolson')

    P_tt = P0_tt
    for i in range(Nt):
        P_tt = fwd_int.solve(P_tt, dT, intervals = 1)

    P = P_tt.full().reshape(N)
    
    err = np.max(np.abs(P-P_ref)) / np.max(np.abs(P_ref))
    err_cn.append(err)
    print('nt ',nt,' error inf ',err)
    
# convergence test
print('Cheby...')
err_ch = []
refinements_ch = [2,4,6,8,10,12,14,16,18,20,22,24,28,32]
for nt in refinements_ch:
    
    fwd_int = ttInt(A_tt, epsilon = 1e-14, N_max = nt, dt_max = 1000.0,method='cheby')

    P_tt = P0_tt
    for i in range(Nt):
        P_tt = fwd_int.solve(P_tt, dT, intervals = 1)
        P_tt = P_tt.round(1e-14)

    P = P_tt.full().reshape(N)
    
    err = np.max(np.abs(P-P_ref)) / np.max(np.abs(P_ref))
    err_ch.append(err)
    print('nt ',nt,' error inf ',err)
    
# convergence test
print('Legendre...')
err_le = []
refinements_le = [2,4,6,8,10,12,14,16,18,20,22,24,28,32]
for nt in refinements_le:
    
    fwd_int = ttInt(A_tt, epsilon = 1e-14, N_max = nt, dt_max = 1000.0,method='legendre')

    P_tt = P0_tt
    for i in range(Nt):
        P_tt = fwd_int.solve(P_tt, dT, intervals = 1)
        P_tt = P_tt.round(1e-14)
    P = P_tt.full().reshape(N)
    
    err = np.max(np.abs(P-P_ref)) / np.max(np.abs(P_ref))
    err_le.append(err)
    print('nt ',nt,' error inf ',err)
    
# convergence test
print('Epsilon of the solver...')
err_eps = []
refinements_epsilon = 10.0 ** (-np.arange(1,11))
for eps in refinements_epsilon:
    
    fwd_int = ttInt(A_tt, epsilon = eps, N_max = 16, dt_max = 1000.0,method='cheby')

    P_tt = P0_tt
    for i in range(Nt):
        P_tt = fwd_int.solve(P_tt, dT, intervals = 1)

    P = P_tt.full().reshape(N)
    
    err = np.max(np.abs(P-P_ref)) / np.max(np.abs(P_ref))
    err_eps.append(err)
    print('epsilon ',eps,' error inf ',err)
    

print('Epsilon vs Nt ...')
refinements_epsilon_2 = 10.0 ** (-np.arange(1,13))
refinements_ch2 = [2,3,4,5,6,7,8]
err_eps_ch = []
for eps in refinements_epsilon_2:
    err_temp = []
    for nt in refinements_ch2:
        fwd_int = ttInt(A_tt, epsilon = eps, N_max = nt, dt_max = 1000.0,method='cheby')
    
        P_tt = P0_tt
        for i in range(Nt):
            P_tt = fwd_int.solve(P_tt, dT, intervals = 1)
    
        P = P_tt.full().reshape(N)
        
        err = np.max(np.abs(P-P_ref)) / np.max(np.abs(P_ref))
        err_temp.append(err)
        print('epsilon ',eps,' nt ',nt,' error inf ',err)
    err_eps_ch.append(err_temp)
    
    
    
#%% plots
import tikzplotlib
plt.figure()
plt.loglog(refinements_implicit,err_implicit)
plt.loglog(refinements_cn[:-1],err_cn[:-1])
plt.loglog(refinements_ch[:],err_ch[:])
# plt.loglog(refinements_le[:],err_le[:])
plt.xlabel(r'$N_t$')
plt.ylabel(r'max relative error')
plt.grid()
plt.legend(['Implicit Euler','Crank-Nicolson','Chebyshev'])
tikzplotlib.save('convergence_Nt.tex')

# plt.figure()
# plt.loglog(dT/np.array(refinements_implicit),np.array(err_implicit))
# plt.loglog(dT/np.array(refinements_cn)[:-1],np.array(err_cn)[:-1])
# plt.xlabel(r'$\Delta t$ [s]')
# plt.ylabel(r'max relative error')
# plt.grid()
# plt.legend(['Implicit Euler','Crank-Nicolson'])
# tikzplotlib.save('convergence_dt.tex')

plt.figure()
plt.loglog(refinements_epsilon,err_eps)
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'max relative error')
plt.grid()
tikzplotlib.save('convergence_eps.tex')

plt.figure()
plt.loglog(dT/np.array(refinements_ch2),np.array(err_eps_ch).transpose())
plt.xlabel(r'$\Delta t$ [s]')
plt.ylabel(r'max relative error')
plt.legend([r'$\epsilon=$'+str(eps) for eps in refinements_epsilon_2])
plt.grid()
tikzplotlib.save('convergence_eps_multiple.tex')

plt.figure()
plt.loglog(np.array(refinements_epsilon_2),np.array(err_eps_ch))
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'max relative error')
plt.legend([r'$T=$'+str(tmp)+'' for tmp in np.array(refinements_ch2)])
plt.grid()
tikzplotlib.save('convergence_Nt_multiple.tex')