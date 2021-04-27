# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 23:48:12 2019

@author: ion
"""
import numpy as np
import scipy.sparse as sps
import timeit
#import tensorflow as tf
import numba
from tt_extra import mat_to_tt
import matplotlib.pyplot as plt 
import tt

def nz_prod(l, i):
    tmp = 1.0
    for (k, n) in zip(l, i):
        if n != 0:
            tmp *= k
    return tmp

@numba.jit('float64[:](int64[:,:],int64[:])',nopython=True)
def Propensity(M,x):
    props = np.ones(M.shape[0],dtype=np.float64)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]!=0:
                props[i] *= M[i,j]*x[j]
    return props


# @numba.jit('Tuple((float64[:],int64[:,:]))(int64[:],float64,int64[:,:],int64[:,:],float64[:])',nopython=True)
def Gillespie(X0,tmax,Pre,nu,C,props = None):
   
    num_r = Pre.shape[0]
    num_s = Pre.shape[1]

            
    t = [0.0]
    xs = [ list(X0.copy()) ] 
    reaction_indices = []
    x = X0.copy()
    
    total_time = 0.0
    while True:
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        # print(Pre.shape,x.shape)
        if props == None:
            a = C * Propensity(Pre, x)
        else:
            a = np.array([p(x.reshape([1,-1])) for p in props]).flatten() 
            # print(x,a)
            a = C*a
        a0 = np.sum(a)    
       
        if a0 != 0:
            treact = -np.log(r1)/a0
            cum = np.cumsum(a/a0)
                
            choice = num_r-1
            for k in range(num_r):
                if cum[k]>r2:
                    choice = k
                    break
            
            total_time += treact
            if total_time > tmax:
                t.append(tmax)
                xs.append(list(x))
                break
            else:
                x += nu[choice,:]
                
                reaction_indices.append(choice)
                xs.append(list(x))
                t.append(total_time)
        else: 
            t.append(tmax)
            xs.append(list(x))
            break
    return np.array(t),np.array(xs),np.array(reaction_indices)

def Observations_grid(time_grid,reaction_time,reaction_jumps):
    
    observation = np.zeros((time_grid.size,reaction_jumps.shape[1]),dtype=np.int64)
    
    index = 1
    
    for i in range(time_grid.size):    
        while time_grid[i] >= reaction_time[index] and index < reaction_time.size-1:
            index += 1
        
        observation[i,:] = reaction_jumps[index-1]
        
    return observation
    
@numba.jit('int64[:,:,:](int64[:,:],int64,float64[:],int64[:,:],int64[:,:],float64[:])',parallel=False,nopython=True)
def GillespieMultiple(X0,Ns,time,Pre,nu,C):
   
    num_r = Pre.shape[0]
    num_s = Pre.shape[1]
    Nt = time.shape[0]
    tmax = np.max(time)
    
    xs = np.zeros((Nt,nu.shape[1],Ns),dtype=np.int64)
    
    
    for i in numba.prange(Ns):
        # print(i)
        
        x = X0[i,:]
            
        counter = 1
        state_temp = np.zeros((Nt,nu.shape[1]),dtype=np.int64)
        state_temp[0,:] = x
        
        total_time = 0.0
        while total_time<=tmax:
            
            
                
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            a = C * Propensity(Pre, x)
            
            a0 = np.sum(a)    
           
            if a0 != 0:
                treact = np.abs(-np.log(r1)/a0)
                
                cum = np.cumsum(a/a0)
                    
                choice = num_r-1
                for k in range(num_r):
                    if cum[k]>r2:
                        choice = k
                        break
                    
                while counter < Nt and time[counter] < total_time :
                    state_temp[counter,:] = x.copy()
                    counter += 1
                    
                total_time += treact
                # print (total_time,treact,a,choice,nu[choice,:],x)
                
                if total_time <= tmax:
                    x += nu[choice,:]

            else: 
                total_time = tmax+1
                
        while counter < Nt and time[counter] <= total_time :
            state_temp[counter,:] = x.copy()
            counter += 1    
    
        xs[:,:,i] = state_temp
                
    return xs

def CompleteObservations(sample,time,indices,props,Cs,alpha_prior=1.0,beta_prior=0.0):
    
    nr = Cs.size
    n = time.size
    T = time.max()
    
    N = np.array([np.sum(indices==j) for j in range(nr)])
       
    # print(N)
    
    G = N*0
    
    for j in range(nr):
        for k in range(1,n-1):
            G[j] += props[j](sample[k-1,:].reshape([1,-1]))[:] * (time[k]-time[k-1])
        G[j] += props[j](sample[-1,:].reshape([1,-1])) * (T - time[n-1])
    
    alpha_posterior = alpha_prior + N
    beta_posterior = beta_prior + G
    
    return alpha_posterior, beta_posterior
    
    
class CME():

    num_states: int

    def __init__(self, Ns, Pre, Post, C, Props=None):
        self.Pre = Pre
        self.size = Ns
        self.Post = Post
        self.nu = Post - Pre
        self.num_states = np.prod(Ns)
        self.num_r = C.size
        self.Props = Props
        self.C = C

    def construct_generator(self):

        idx_row = []
        idx_col = []
        vals = []
        t = timeit.time.time()
        if self.Props is None:

            for i in range(self.num_states):
                xk = np.unravel_index(i, self.size)

                for k in range(self.num_r):
                    # enter diagonal
               
                    xp = xk - self.nu[k,:]
                    # print(k,xk,xp)
                


                    if all(xp >= 0) and all(xp < self.size):

                        idx_row.append(i)
                        idx_col.append(i)
                        vals.append(-self.C[k] * nz_prod(xk * self.Pre[k, :], self.Pre[k, :]))

                        idx_row.append(i)
                        idx_col.append(np.ravel_multi_index(xp, self.size))
                        vals.append(self.C[k] * nz_prod(xp * self.Pre[k, :], self.Pre[k, :]))
        else:

            for i in range(self.num_states):
                xk = np.unravel_index(i, self.size)

                for k in range(self.num_r):
                    # enter diagonal
                    pass

            # I = list(range(self.num_states))
            # Xk = np.array(np.unravel_index(np.arange(self.num_states),self.size)).transpose()

            # for k in range(self.num_r):
            #     idx_row += (I)
            #     idx_col += (I)
            #     vals += list(-self.C[k]*self.Props[k](Xk))

            #     Xp = Xk - self.nu[k,:]

            #     idx_keep = np.logical_and(np.all(Xp>=0,axis=1), np.all(Xp<self.size,axis=1))

            #     tmp_row = np.arange(self.num_states) 
            #     tmp_row = tmp_row[idx_keep]

            #     Xp = Xp[idx_keep,:]

            #     tmp_col = np.ravel_multi_index(Xp.transpose(),self.size)
            #     tmp_val = self.C[k]*self.Props[k](Xp)

            #     idx_row += list(tmp_row)
            #     idx_col += list(tmp_col)
            #     vals += list(tmp_val)

            # for i in range(self.num_states):
            #     xk = np.unravel_index(i,self.size)

            #     for k in range(self.num_r):
            #         # enter diagonal
            #         idx_row.append(i)
            #         idx_col.append(i)

            #         vals.append(-self.C[k]*self.Props[k](xk))

            #         xp = xk - self.nu[k,:]

            #         if all(xp>=0) and all(xp<self.size):
            #             idx_row.append(i)
            #             idx_col.append(np.ravel_multi_index(xp,self.size))
            #             vals.append(self.C[k]*self.Props[k](xp))
      #  print( timeit.time.time() - t )     
        t = timeit.time.time()           
        self.gen = sps.bsr_matrix((np.array(vals), (np.array(idx_row), np.array(idx_col))), shape=(self.num_states, self.num_states))
     #   print( timeit.time.time() - t )   
    def construct_generator2(self,to_tf=False,save_list=False):
        idx_row = None
        idx_col = None
        vals = None

        I = list(range(self.num_states))
        Xk = np.array(np.unravel_index(np.arange(self.num_states), self.size)).transpose()

        for k in range(self.num_r):
            Xp = Xk + self.nu[k, :]
            idx_keep = np.logical_and(np.all(Xp >= 0, axis=1), np.all(Xp < self.size, axis=1))

            # print(Xk)
            # print(Xp)
            # add diagonal 
            tmp = np.arange(self.num_states)
            tmp = tmp[idx_keep]
            if idx_row is None:
                idx_row = tmp
                idx_col = tmp
            else:
                idx_row = np.concatenate((idx_row,tmp))
                idx_col = np.concatenate((idx_col,tmp))
                
            tmp = Xk[idx_keep, :]

            if vals is None:
                vals = (-self.C[k] * self.Props[k](tmp))
            else:
                vals = np.concatenate((vals,(-self.C[k] * self.Props[k](tmp))))
                
            # add non diagonal
            tmp_col = np.arange(self.num_states)
            tmp_col = tmp_col[idx_keep]

            Xp = Xp[idx_keep, :]

            tmp_row = np.ravel_multi_index(Xp.transpose(), self.size)
            tmp_val = self.C[k] * self.Props[k](Xk[idx_keep, :])
            # print(tmp_row)
            # print(tmp_col)   
            # print(tmp_val)
            if idx_row is None:
                idx_row = tmp_row
                idx_col = tmp_col
                vals = tmp_val
            else:
                idx_row = np.concatenate((idx_row,tmp_row))
                idx_col = np.concatenate((idx_col,tmp_col))
                vals = np.concatenate((vals,tmp_val))
            
            #print(np.array(vals), np.array(idx_row), np.array(idx_col))
            
        if save_list :
            self.gen = (np.array(idx_row),np.array(idx_col),np.array(vals))
            
        elif not to_tf:
            vals = np.array(vals)
            idx_row = np.array(idx_row)
            idx_col = np.array(idx_col)
            
            self.gen = sps.csr_matrix((vals, (idx_row, idx_col)), shape=(self.num_states, self.num_states))
            idx_row = None
            idx_col = None
            vals = None
        else:
            pass
#            indices =  tf.transpose(tf.constant([idx_row,idx_col],dtype=tf.int64))
#            values = tf.constant(vals,dtype= tf.float64)
#            self.gen = tf.sparse.SparseTensor(indices, values, dense_shape=(self.num_states, self.num_states))
    
   
    def apply_cme_operator(self,v):
        
        res = v.copy()*0
        
        I = list(range(self.num_states))
        Xk = np.array(np.unravel_index(np.arange(self.num_states), self.size)).transpose()

        for k in range(self.num_r):
            Xp = Xk + self.nu[k, :]
            idx_keep = np.logical_and(np.all(Xp >= 0, axis=1), np.all(Xp < self.size, axis=1))

            # print(Xk)
            # print(Xp)
            # add diagonal 
            tmp = np.arange(self.num_states)
            tmp = tmp[idx_keep]
            idx_row = (tmp)
            idx_col = (tmp)

            tmp = Xk[idx_keep, :]

            vals = (-self.C[k] * self.Props[k](tmp))
            
            res[idx_row] += vals * v[idx_col]
            
            # add non diagonal
            tmp_col = np.arange(self.num_states)
            tmp_col = tmp_col[idx_keep]

            Xp = Xp[idx_keep, :]

            tmp_row = np.ravel_multi_index(Xp.transpose(), self.size)
            tmp_val = self.C[k] * self.Props[k](Xk[idx_keep, :])
            # print(tmp_row)
            # print(tmp_col)   
            # print(tmp_val)
            res[idx_row] += tmp_val * v[idx_col]
            
        return res
    
    def CME_tt(self,fs,C,nu):
        
        d = len(self.size)
        
        A1 = []
        A2 = []
        
        for i in range(d):
            
            
            
            core = np.zeros((self.size[i],self.size[i]))
            for k in range(self.size[i]):
                core[k,k] = fs[i](k) if  k+nu[i]>=0 and k+nu[i]<self.size[i] else 0.0
            A1.append(core.reshape([1,self.size[i],self.size[i],1]))
            # print(core[-10:,-10:])
            
            core = np.zeros((self.size[i],self.size[i]))
            for k in range(self.size[i]):
                if k+nu[i]>=0 and k+nu[i]<self.size[i]:
                    core[k+nu[i],k] = fs[i](k)
            A2.append(core.reshape([1,self.size[i],self.size[i],1]))
            # print(core[-10:,-10:])
            
        Att =C*(tt.matrix().from_list(A2) - tt.matrix().from_list(A1))

        Att = Att.round(1e-18,64)
            
        return Att                    
            
            
    def construct_generator_tt(self,as_list = False):
        
        if as_list:
            Att = []
        else:
            Att = tt.eye(self.size) * 0
        
        for i in range(self.num_r):
            
            fs = []
            
            for k in range(len(self.size)):
                if self.Pre[i,k] != 0:
                    fs.append((lambda a: lambda x: x*a)(self.Pre[i,k]))
                else: 
                    fs.append(lambda x : 1.0)
                    
            A = self.CME_tt(fs,self.C[i],self.nu[i,:])
                    
            if as_list:
                Att.append(A)
            else:
                Att = Att + A
                Att = Att.round(1e-12)
            
        return Att           
            
#    def tt_operator(self,eps,rmax=32):
#        cores = mat_to_tt(self.gen,self.size,self.size,eps,rmax,tf_sparse=True)
#        self.gen_tt = t3f.TensorTrain(cores)

    @numba.jit(parallel=True)
    def EmpiricalPMF(self,sample):
        PMF = np.zeros(self.size)
        ns = sample.shape[1]
        
        for i in range(ns):
            PMF[tuple(sample[:,i])] += 1
            
        PMF = PMF/np.sum(PMF)
        return PMF
        
    def ssa(self,x0,time,Ns = 1):
        if x0.ndim==1 :
            x0 = np.tile(x0.reshape([-1,1]),Ns).transpose()
        
        Sample = GillespieMultiple(x0.astype(np.int64),Ns,time.astype(np.float64), self.Pre.astype(np.int64), self.nu.astype(np.int64), self.C.astype(np.float64))
        return Sample
    
    
    
# # define reaction 
# rates = np.array([5e-4,1e-4,1e-4,5e-4])
# Pre =np.array( [[1,0],[1,1],[1,1],[0,1]])
# Post = np.array([[2,0],[0,1],[1,2],[0,0]])
# Props = [ lambda x: x[:,0] , lambda x: x[:,0]*x[:,1]  , lambda x: x[:,0]*x[:,1]  , lambda x: x[:,1] ]

# x = np.array([20,5],dtype=np.int64)
# time = np.linspace(0,1000,100)
# mdl = CME(2,Pre,Post,rates,Props)
# sample = mdl.ssa(x,time,4)

# # Time,Sample = Gillespie(x.astype(np.int64),1000.0, Pre.astype(np.int64), nu.astype(np.int64), rates.astype(np.float64))

# plt.plot(time,sample[:,:,0])
# time = np.linspace(0,1000.0,1000)
# tme = timeit.time.time()
# Sample = GillespieMultiple(x,10000,time, Pre, nu, rates)
# tme = timeit.time.time() - tme

# plt.figure()
# plt.plot(time,Sample[:,:,1])
