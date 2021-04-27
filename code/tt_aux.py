#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:01:59 2021

@author: yonnss
"""

import numpy as np
import tt

def tt2qtt(T):
    
    N = []
    for n in T.n: N += int(np.log2(n))*[2]
    T_qtt = tt.reshape(T,N)
    T_qtt.ps =T_qtt.ps.astype(np.int32)
    return T_qtt

def qtt2tt(T_qtt,N):
    T = tt.reshape(T_qtt,N)
    T.ps =T.ps.astype(np.int32)
    return T

def ttm2qttm(T):
    
    N = []
    for n in T.n: N += int(np.log2(n))*[2]
    T_qtt = tt.reshape(T,np.array([N,N]).transpose())
    T_qtt.tt.ps =T_qtt.tt.ps.astype(np.int32)
    return T_qtt


def mode_prod(t,Ms):
    cores = t.to_list(t)
    for i in range(len(cores)):
        cores[i] = np.einsum('ijk,jl->ilk',cores[i],Ms[i])
    return tt.tensor().from_list(cores)    

def tt_meshgrid(xs):
    
    lst = []
    for i  in range(len(xs)):
        lst.append(tt.mkron([tt.tensor(xs[i]) if i==k else tt.ones([xs[k].size]) for k in range(len(xs)) ]))
    return lst

def tt_size(tensor):
    return sum([c.size for c in tensor.to_list(tensor)])