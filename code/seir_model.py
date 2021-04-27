#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:31:30 2020

@author: ion
"""




import tensorflow as tf
import t3f
import numpy as np
import matplotlib.pyplot as plt
from CME import CME,Gillespie
import timeit
import scipy.integrate
import numba
import scipy.sparse
from tt_extra import mat_to_tt
import tt
import tt.amen
import tt.eigb
from ttInt import ttInt


#%% REaction cooefficeints
# define reaction 
rates = np.array([0.1,0.5,1.0,0.01,0.01,0.01,0.4])
Pre =np.array( [[1,0,1,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])
Post = np.array([[0,1,1,0],[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1],[1,0,0,0]])
Props = [ lambda x: x[:,0]*x[:,2] , lambda x: x[:,1]  , lambda x: x[:,2]  , lambda x: x[:,0] , lambda x: x[:,1] , lambda x: x[:,2] , lambda x: x[:,0]*0+1 ]


# construct the model and the CME operator
N = 4*[80] # state truncation
mdl = CME(N, Pre,Post,rates,Props) # model
# mdl = CME(N, np.array( [[1,0,1,0]]),np.array([[0,1,1,0]]), np.array([0.1]),[ lambda x: x[:,0]*x[:,2]])
Ns = 20000

x0 = np.array([50,4,0,0])
sigma = 1.0
p1 = np.exp(-0.5*(x0[0]-np.arange(N[0]))**2/sigma)
ch1 = np.random.choice(np.arange(N[0]),(Ns,1),p=p1/np.sum(p1))
p2 = np.exp(-0.5*(x0[1]-np.arange(N[1]))**2/sigma)
ch2 = np.random.choice(np.arange(N[1]),(Ns,1),p=p2/np.sum(p2))
p3 = np.exp(-0.5*(x0[2]-np.arange(N[2]))**2/sigma)
ch3 = np.random.choice(np.arange(N[2]),(Ns,1),p=p3/np.sum(p3))
p4 = np.exp(-0.5*(x0[3]-np.arange(N[3]))**2/sigma)
ch4 = np.random.choice(np.arange(N[3]),(Ns,1),p=p4/np.sum(p4))

#%% Monte Carlo
# time scale
Nt = 70
dT = 10.0/100
time_sample = np.arange(Nt+1) * dT
# draw sample
sample = mdl.ssa(np.concatenate((ch1,ch2,ch3,ch4),axis=1), time_sample,Ns )

# plot sample
plt.figure()
plt.title('Sample')
plt.plot(time_sample,sample[:,0,0],'b')
plt.plot(time_sample,sample[:,1,0],'orange')
plt.plot(time_sample,sample[:,2,0],'r')
plt.plot(time_sample,sample[:,3,0],'g')
plt.legend(['Susceptible','Exposed','Infected','Recovered'])
plt.ylabel(r'#individuals')
plt.xlabel(r'$t$ [d]')

# plot sample
plt.figure()
plt.title('Means')
plt.plot(time_sample,np.mean(sample[:,0,:],1),'b')
plt.plot(time_sample,np.mean(sample[:,1,:],1),'orange')
plt.plot(time_sample,np.mean(sample[:,2,:],1),'r')
plt.plot(time_sample,np.mean(sample[:,3,:],1),'g')
plt.legend(['Susceptible','Exposed','Infected','Recovered'])
plt.ylabel(r'#individuals')
plt.xlabel(r'$t$ [d]')

#%% Integrate ODE
A_tt = mdl.construct_generator_tt()



# A_tt = tt.reshape(A_tt,np.array(2*[[15]*8]).transpose())

P = tt.kron(tt.kron(tt.tensor(p1),tt.tensor(p2)),tt.kron(tt.tensor(p3),tt.tensor(p4)))
P = P * (1/tt.sum(P))
P0 = P
# P = tt.reshape(P,[15]*8)

x_S = tt.kron(tt.tensor(np.arange(N[0])),tt.ones(N[1:]))
x_E = tt.kron(tt.ones([N[0]]),tt.kron(tt.tensor(np.arange(N[1])),tt.ones(N[2:])))
x_I = tt.kron(tt.ones(N[:2]),tt.kron(tt.tensor(np.arange(N[2])),tt.ones([N[3]])))
x_R = tt.kron(tt.ones(N[:3]),tt.tensor(np.arange(N[3])))

epsilon = 1e-10
rmax = 30


#%% reference ode solution
# print('Reference...')
# tme_ode45 = timeit.time.time()
# mdl.construct_generator2(to_tf=False)
# Gen = mdl.gen
# def func(t,y):
#     print(t)
#     return Gen.dot(y)

# # solve CME
# print('ODE solver...')
# res = scipy.integrate.solve_ivp(func,[0,Nt*dT],P0.full().flatten(),t_eval=[0,Nt*dT])
# Pt = res.y.reshape(N+[-1])
# tme_ode45 = timeit.time.time() - tme_ode45

# P_ref = Pt[:,:,:,:,-1]

print('Loading reference....')
P_mc = np.load('./reference_ode.dat',allow_pickle = True)
P_ref = P_mc 

#%% TT
print('TT integration...')
fwd_int = ttInt(A_tt, epsilon = 1e-6, N_max = 64, dt_max = 1e-1,method='crank–nicolson')


time = 0.0
tme_total = timeit.time.time()
Pms_SE = []
Pms_EI = []
for i in range(Nt):

    
    tme = timeit.time.time()
    
    P = fwd_int.solve(P, dT, intervals = 4)
    
    tme = timeit.time.time() - tme
   
    P = P.round(1e-10,100)
    P = P * (1/tt.sum(P))
    time += dT
    
    Pms_SE.append(tt.sum(tt.sum(P,3),2).full())
    Pms_EI.append(tt.sum(tt.sum(P,0),2).full())

    print('k = ',i,'/',Nt,' at time ',time, ' rank ',P.r,' time ',tme)
    
tme_total = timeit.time.time()-tme_total

# print('TT time ',tme_total,' vs ODE solver time ',tme_ode45)

Pend = P.full()


residual = (Pend-P_ref)[:60,:60,:60,:60]
# residual = residual[:40,:40,:40,:40]
print('Mean rel error ',np.mean(np.abs(residual))/np.max(np.abs(Pend)))
print('Max rel error ',np.max(np.abs(residual))/np.max(np.abs(Pend)))

P_ref[66:,:,:,:] = 0
P_ref[:,66:,:,:] = 0
P_ref[:,:,66:,:] = 0
P_ref[:,:,:,66:] = 0


# P = tt.reshape(P,N)

# P1_end = np.zeros((N[0]))
# P2_end = np.zeros((N[1]))
# P3_end = np.zeros((N[2]))
# P4_end = np.zeros((N[3]))

# for i in range(Ns):
#     P1_end[sample[-1,0,i]] += 1
#     P2_end[sample[-1,1,i]] += 1
#     P3_end[sample[-1,2,i]] += 1
#     P4_end[sample[-1,3,i]] += 1
    
# P1_end = P1_end / np.sum(P1_end)
# P2_end = P2_end / np.sum(P2_end)
# P3_end = P3_end / np.sum(P3_end)
# P4_end = P4_end / np.sum(P4_end)

# P1_end_tt = tt.sum(tt.sum(tt.sum(P,1),1),1)
# P2_end_tt = tt.sum(tt.sum(tt.sum(P,0),1),1)
# P3_end_tt = tt.sum(tt.sum(tt.sum(P,0),0),1)
# P4_end_tt = tt.sum(tt.sum(tt.sum(P,0),0),0)


# plt.figure()
# plt.plot(np.arange(N[0]),P1_end)
# plt.plot(np.arange(N[0]),P1_end_tt.full())
# plt.title('Marginal PMF for S')

# plt.figure()
# plt.plot(np.arange(N[1]),P2_end)
# plt.plot(np.arange(N[1]),P2_end_tt.full())
# plt.title('Marginal PMF for E')

# plt.figure()
# plt.plot(np.arange(N[2]),P3_end)
# plt.plot(np.arange(N[2]),P3_end_tt.full())
# plt.title('Marginal PMF for I')

# plt.figure()
# plt.plot(np.arange(N[3]),P4_end)
# plt.plot(np.arange(N[3]),P4_end_tt.full())
# plt.title('Marginal PMF for R')




    
# plt.figure()
# time = 0.0
# for img in Pms_SE:
#     plt.clf()
#     plt.imshow(img,origin='lower')
    
#     plt.axis('equal')
#     plt.xlabel('Susceptible')
#     plt.ylabel('Exposed')
#     plt.colorbar()
#     plt.pause(0.1)
#     time+=dT
#     print(time)
    
# # Pref = P.full().reshape([-1,1])

# time_total = 0
# Exp = x0.reshape([1,-1])

# import sys
# sys.exit()<0):
    
#     tme = timeit.time.time()
    
#     h = time[i] - time[i-1] 
    
#     tme = timeit.time.time()
#     k1 = tt.matvec(Att,P).round(epsilon,rmax)
#     k2 = tt.matvec(Att,( P + 0.5 * h * k1 ).round(epsilon,rmax)).round(epsilon,rmax)
#     k3 = tt.matvec(Att,( P + 0.5 * h * k2 ).round(epsilon,rmax)).round(epsilon,rmax)
#     k4 = tt.matvec(Att,( P + h * k3 ).round(epsilon,rmax)).round(epsilon,rmax)

#     P = (P + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4).round(epsilon,rmax)

#     P = P*(1/tt.sum(P))
    
 
    
#     time_total += h
    
#     tme = timeit.time.time() - tme
    
    
#     E = np.array([[tt.sum(P*x_S),tt.sum(P*x_E),tt.sum(P*x_I),tt.sum(P*x_R)]])
#     Exp = np.concatenate((Exp,E))
    
#     print('%4d/%4d time %5.3f s' %(i+1,Nt,tme),' rank ',P.r,flush = True)
    
# plt.figure()
# plt.title('Means')
# plt.plot(time[:100],Exp)

# plt.legend(['S','E','I','R'])

#%% MC
# print('Monte Carlo....')
# N1 = 100000
# N2 = 100

# P_mc = np.zeros(N)
# for i in range(N2):
#     print(i)
#     sample = mdl.ssa(np.concatenate((ch1,ch2,ch3,ch4),axis=1), np.arrat([0,Nt*dT]),N1 )
#     for k in range(N1):
#         P_mc[sample[-1,0,k],sample[-1,1,k],sample[-1,2,k],sample[-1,3,k]] += 1

# P_mc = P_mc / np.sum(P_mc)
        
#%% plots
plt.figure()
plt.imshow(Pend.sum(2).sum(2).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('./plots/SE_marginal.tex')

plt.figure()
plt.imshow((Pend-P_ref).sum(2).sum(2).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('./plots/SE_marginal_err.tex')

plt.figure()
plt.imshow(Pend.sum(0).sum(2).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal.tex')

plt.figure()
plt.imshow((Pend-P_ref).sum(0).sum(2).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('./plots/EI_marginal_err.tex')

plt.figure()
plt.imshow(Pend.sum(0).sum(0).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_3$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/IR_marginal.tex')

plt.figure()
plt.imshow((Pend-P_ref).sum(0).sum(0).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_3$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/IR_marginal_err.tex')

plt.figure()
plt.imshow(Pend.sum(1).sum(1).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/SR_marginal.tex')

plt.figure()
plt.imshow((Pend-P_ref).sum(1).sum(1).transpose(),origin='lower')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_4$')
tikzplotlib.save('./plots/SR_marginal_err.tex')



