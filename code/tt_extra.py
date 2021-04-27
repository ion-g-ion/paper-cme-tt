#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:06:20 2020

@author: ion
"""

#import tt
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import tensorflow as tf
import t3f
import timeit
# from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
# import tt

def round_cuda(tt_cores,eps,rmax=100):
    
    N = [c.shape[1] for c in tt_cores]
    d = len(tt_cores)
    r = [1] + [c.shape[2] for c in tt_cores]
    
    if isinstance(rmax,int):
        rmax = [1]+(d-1)*[rmax]+[1]
    
    core = tt_cores[0]
    
    norm = np.zeros(d)
    
    # Orthogonalize left to right
    
    for i in range(d-1):
        core = core.reshape([r[i]*N[i],-1])
        
        core, R = np.linalg.qr(core)
        norm[i+1] = np.linalg.norm(R,ord = 'fro')
        
        if norm[i+1] != 0:
            R = R / norm[i+1]
            
        core_next = tt_cores[i+1].reshape([r[i+1],-1])
        core_next = R @ core_next
        
        r[i+1] = core.shape[1]
        
        tt_cores[i] = core
        tt_cores[i+1] = core_next
        
        core = core_next
        
        
    nrm = norm / np.sqrt(d-1)
    
    core = tt_cores[-1]
    ep=eps/np.sqrt(d-1)
    
    for i in range(d-1,0,-1):
        
        core_prev = tt_cores[i-1]
        
        core = core.reshape([r[i],-1])
        core_prev = core_prev.reshape([-1,r[i]])
    
        
        
        # Use SVD now
        U,S,V = np.linalg.svd(core)
        
        r_chop = rank_chop(S,np.linalg.norm(S)*ep)
        
        r_chop = min([r_chop,rmax[i]])
        
        
        U = U[:,:r_chop]
        S = S[:r_chop]
        V = V[:r_chop,:]
        
        r[i] = r_chop
     
        U = U @ np.diag(S)
        core_prev = core_prev @ U
        core = V
        
        tt_cores[i] = core
        tt_cores[i-1] = core_prev
        # print(i)
        # print(core.shape)
        # print(core_prev.shape)
        core = core_prev
    # print([c.shape for c in tt_cores])    
    core0 = tt_cores[0]
    nrm[0] = np.linalg.norm(core0,ord='fro')
    # print(nrm[0])
    if not nrm[0] == 0:
        core0 = core0 / nrm[0]
        tt_cores[0] = core0
    
    
    nrm = np.exp(np.sum(np.log(np.abs(nrm)))/d)
    # print([c.shape for c in tt_cores])
    for i in range(d):
        tt_cores[i] = np.reshape(tt_cores[i] * nrm,[r[i],N[i],r[i+1]])
        

    return tt_cores, r        
     
    
    
def sum_ind(TT,ind):
    for i in ind:
        TT = tt.sum(TT,axis = i)
    return TT
    
def mat_to_tt(A,M,N,eps,rmax = 100,tf_sparse=False):
    
    d = len(M)
    if len(M)!=len(N):
        raise('Dimension mismatch')
        return
    
    if tf_sparse:
        A = tf.sparse.reshape(A,M+N)
        # print(A.shape)
        permute = np.arange(2*d).reshape([2,d]).transpose().flatten()
        # print(permute)
        A = tf.sparse.transpose(A,permute)
        # print(A.shape)
        
        A = tf.sparse.reshape(A,[i[0]*i[1] for i in zip(M,N)])
        # print(A.shape)
        ttv = to_tt(A,eps=eps,rmax=rmax,tf_sparse=True)
    else:
        A = A.reshape(M+N)
        # print(A.shape)
        permute = np.arange(2*d).reshape([2,d]).transpose().flatten()
        # print(permute)
        A = A.transpose(permute)
        # print(A.shape)
        
        A = A.reshape([i[0]*i[1] for i in zip(M,N)])
        # print(A.shape)
        ttv = to_tt(A,eps=eps,rmax=rmax)
    
    # cores = tt.tensor.to_list(ttv)
    
    cc = []
    
    for i in range(d):
        # print(cores[i].shape)
        tmp = np.moveaxis(ttv[i], [0,1,2], [1,0,2])
        # print(tmp.shape)
        tmp = tmp.reshape([M[i],N[i],tmp.shape[1],tmp.shape[2]])
        # print(tmp.shape)
        tmp = np.moveaxis(tmp,[2,0,1,3],[0,1,2,3])
        # print(tmp.shape)
        cc.append(tmp)
        
   # ttm = tt.matrix.from_list(cc)

    return cc

def rank_chop(s,eps):
    if np.linalg.norm(s) == 0.0:
        return 1
    
    if eps <= 0.0:
        return s.size
    
    R = s.size-1
    
    while R>0:
        if np.sum(s[R:]**2) >= eps**2:
            break;
        R -= 1
    
    return R+1
    
def to_tt(A,N=None,eps=1e-14,rmax=100,tf_sparse=False):
    if N == None:
        N = list(A.shape)
      
    
        
    d = len(N)
    r = [1]*(d+1)
    if not isinstance(rmax,list):
        rmax = [1] + (d-1)*[rmax] + [1]
        
    C = A
   
    cores = []
    
    pos = 0;
    
    ep = eps/np.sqrt(d-1)
    
    if tf_sparse:
        C = tf.sparse.reshape(C,[N[0]*r[0],-1])
        idx_row = C.indices[:,0].numpy()
        idx_col = C.indices[:,1].numpy()
        valz = C.values.numpy()
        C = sps.csc_matrix((np.array(valz), (np.array(idx_row), np.array(idx_col))), shape=(C.shape[0], C.shape[1]))
        
    for i in range(d-1):
        
        m = N[i]*r[i]
        
        C = C.reshape([m,-1])
        print(C.shape)
            
        if scipy.sparse.issparse(C):
            print('SP',type(C))
            ttt = timeit.time.time()
            if not scipy.sparse.isspmatrix_csc(C):
                C = scipy.sparse.csc_matrix(C)
                

            u, s, v = scipy.sparse.linalg.svds(C,k=np.min([np.min([min(C.shape)])-1,rmax[i+1]]),which='LM')
            # uu,ss,vv = np.linalg.svd(C.toarray(),full_matrices=False)
            # u, s, v = sparsesvd(C, np.min([np.min([min(C.shape)]),rmax[i+1]]))
            IDX = np.argsort(s)[::-1]
            
            
            # print('np: ',ss)
            print(timeit.time.time()-ttt)
            print(u.shape,s.shape,v.shape)
            
            u = u[:, IDX]
            s = s[IDX]
            v = v[IDX, :]
            print('sp ',s)
            # u = uu
            # s = ss
            # v = vv
            # u = u.transpose()
            
        else:
            print('NP')
            u, s, v = scipy.linalg.svd(C,full_matrices=False)
            
        
        # v = v.transpose()
        
        r1 = rank_chop(s, ep*np.linalg.norm(s))
        r1 = np.min([r1,rmax[i+1]])
        u = u[:,:r1]
        s = s[:r1]
        r[i+1] = r1
        
        cores.append(u.reshape([r[i],N[i],r1]))
        
        
        # v = v[:,:r1]
        v = v[:r1,:]
        
               
        v=np.diag(s) @ v
        
        if scipy.sparse.issparse(C):
            v = scipy.sparse.csc_matrix(v)
            
        C = v
        
    if scipy.sparse.issparse(C):
        cores.append(C.toarray().reshape([r[-2],N[-1],-1]))
    else:
        cores.append(C.reshape([r[-2],N[-1],-1]))
    # cores.append(C.flatten())
    # cores = np.concatenate(cores)
    #tensor = tt.tensor().from_list(cores)
    
    return cores
    
# a,b,c,d = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,23),np.linspace(0,2,8),np.linspace(0,5,30))
# A = 1/(a**2+b**2+c**2+d**2+1)
# Att = tt.tensor(A)
# Att_r = Att.round(1e-5,5)
# crz_round, r = round_cuda(Att.to_list(Att), 1e-5, 5)
# Att_rr = tt.tensor().from_list(crz_round)

# print((Att-Att_r).norm()/Att.norm())
# print((Att-Att_rr).norm()/Att.norm())
# print((Att_rr-Att_r).norm()/Att.norm())

# A = np.arange(3*4*5*6).reshape([3*4,5*6])
# A = np.outer(np.arange(3*4),np.arange(5*6))
# corz = mat_to_tt(A,[3,4],[5,6],1e-4)

# i1 = 2
# j1 =1
# i2 = 0
# j2 = 3
# print(sum(corz[0][0,i1,j1,:]*corz[1][:,i2,j2,0]))
# print(A[i1*4+i2,j1*6+j2])


# At3f = t3f.to_tt_matrix(tf.constant(A,dtype=tf.float64), shape=((3,4), (5,6)), epsilon=1e-20, max_tt_rank=10)
# Att = t3f.TensorTrain(corz)

# print(tf.reduce_sum(At3f.tt_cores[0][0,i1,j1,:]*At3f.tt_cores[1][:,i2,j2,0]))
# print(np.sqrt(t3f.frobenius_norm_squared(Att-At3f).numpy()))

# Att = tt.matrix().from_list(corz)
# b = np.outer(np.arange(5),np.arange(6))

# plm = tt.matvec(Att,tt.tensor(b))
# plmref = (A @ b.flatten()).reshape([3,4])
