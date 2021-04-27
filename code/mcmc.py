#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:37:17 2021

@author: yonnss
"""


import tt
import scipy.io
import numpy as np
from CME import CME,Gillespie,CompleteObservations,Observations_grid
import matplotlib.pyplot as plt
import scipy.integrate
import tt.amen
import datetime
import sys
import scipy.interpolate
import scipy.stats
from mpl_toolkits import mplot3d
from ttInt import ttInt
from tt_aux import *
import tensorflow as tf
# import tensorflow_probability as tfp


def eval_post(Atts,params,P0,time_observation,observations,obs_operator,eps=1e-7,method = 'cheby',dtmax=0.1,Nmax = 16):
    
    
    Att = Atts[0]*params[0]
    for i in range(1,params.size):
        Att += Atts[i]*params[i]
    Att = Att.round(1e-12)
    
    qtt = True
    if qtt:
        A_qtt = ttm2qttm(Att)
        integrator = ttInt(A_qtt, epsilon = eps, N_max = Nmax, dt_max = 1.0,method=method)
        P = tt2qtt(P0)
    else:
        integrator = ttInt(Att, epsilon = eps, N_max = Nmax, dt_max = 1.0,method=method)
        P = P0
       
    
    ps = []
    for i in range(1,time_observation.size):
        
        dt = time_observation[i]-time_observation[i-1]
        # print(i,int(np.ceil(dt/dtmax)))
        # tme = timeit.time.time() 
        P = integrator.solve(P, dt, intervals = int(np.ceil(dt/dtmax)),qtt=True)
        Po_tt = obs_operator(observations[i,:],P,time_observation[i]) 
        
        Po_tt = tt2qtt(Po_tt)
        ps.append(tt.dot(Po_tt,P))
        
        P = (P*Po_tt).round(1e-9)
        P = P*(1/ps[-1])
        # tme = timeit.time.time() - tme
        # print(i,' time ',tme,'  ',P.r)
        
    ps = np.array(ps)
    return ps

def eval_post_full(mdl,params,P0,time_observation,observations,obs_operator,eps=1e-7,method = 'cheby',dtmax=0.1,Nmax = 16):
    
    tme = datetime.datetime.now()
    mdl.C = params
    mdl.construct_generator2(to_tf=False)
    # mdl.construct_generator2(to_tf=True)
    
    Gen = mdl.gen
    def func(t,y):
        # print(t)
        return Gen.dot(y)
        # return np.matmul(Gen,y)
        # return tf.sparse.sparse_dense_matmul(Gen,y.reshape([-1,1])).numpy().flatten()
    tme = datetime.datetime.now()- tme
    # print(tme)

    P = P0.full()

       
    tme = datetime.datetime.now()
    ps = []
    for i in range(1,time_observation.size):
        
        dt = time_observation[i]-time_observation[i-1]
        
        # solve CME
        res = scipy.integrate.solve_ivp(func,[0,dt],P.flatten())
        P = res.y[:,-1].reshape(mdl.size)
        
        Po = obs_operator(observations[i,:],P,time_observation[i]).full()
        
        P = P*Po
        Z = np.sum(P)
        ps.append(Z)
        
        
        P = P*(1/Z)
        # tme = timeit.time.time() - tme
        # print(i,' time ',tme,'  ',P.r)
    tme = datetime.datetime.now() - tme 
    # print(tme)
    ps = np.array(ps)
    return ps



def eval_post_tf(mdl,params,P0,time_observation,observations,obs_operator,eps=1e-7,method = 'cheby',dtmax=0.1,Nmax = 16):
    
    # tme = datetime.datetime.now()
    mdl.C = params

    mdl.construct_generator2(to_tf=True)
    
    # Gen = tf.sparse.reorder(mdl.gen)
    Gen = tf.sparse.to_dense(mdl.gen,validate_indices=False)
    
    def func(t,y):
        # print(t)
        # return tf.sparse.sparse_dense_matmul(Gen,y)
        return Gen @ y
    # tme = datetime.datetime.now()- tme
    # print(tme)

    P = tf.constant(P0.full().reshape([-1,1]))

       
    # tme = datetime.datetime.now()
    ps = []
    for i in range(1,time_observation.size):
        print(i)
        dt = time_observation[i]-time_observation[i-1]
        
        # solve CME
        results = tfp.math.ode.DormandPrince().solve(func, 0, P, solution_times=[0, dt])

        P = results.states[1]
        
        Po = tf.reshape(tf.constant(obs_operator(observations[i,:],P,time_observation[i]).full()),[-1,1])
        
        P = P*Po
        Z = tf.reduce_sum(P).numpy()
        ps.append(Z)
        
        
        P = P/Z
        # tme = timeit.time.time() - tme
        # print(i,' time ',tme,'  ',P.r)
    # tme = datetime.datetime.now() - tme 
    # print(tme)
    ps = np.array(ps)
    return ps