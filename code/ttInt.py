#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:40:25 2020

@author: ion
"""
from tt_aux import *
import tt
import tt.amen
import numpy as np
import matplotlib.pyplot
import datetime
from basis import *

       
            
class ttInt():
    
    def  __init__(self, Operator, epsilon = 1e-6, N_max = 64, dt_max = 1e-1, method = 'implicit-euler'):
        
        self.A_tt = Operator
        self.epsilon = epsilon
        self.N_max = N_max
        self.method = method
        
    def get_SP(self,T,N):
        if self.method == 'implicit-euler':
            S = np.eye(N)-np.diag(np.ones(N-1),-1)
            P = T*np.eye(N)/N
            ev = (S@np.ones((N,1))).flatten()
            basis = None
        elif self.method == 'crank–nicolson':
            S = np.eye(N)-np.diag(np.ones(N-1),-1)
            P = np.eye(N)+np.diag(np.ones(N-1),-1)
            P[0,:] = 0
            P = P * T / (2*(N-1))
            ev = (S@np.ones((N,1))).flatten()
            basis = None
        elif self.method == 'cheby':
            basis = ChebyBasis(N,[0,T])
            S = basis.get_stiff()+np.outer(basis(np.array([0])).flatten(),basis(np.array([0])).flatten())
            P = basis.get_mass()
            ev = basis(np.array([0])).flatten()
        elif self.method == 'legendre':
            basis = LegendreBasis(N,[0,T])
            S = basis.get_stiff()+np.outer(basis(np.array([0])).flatten(),basis(np.array([0])).flatten())
            P = basis.get_mass()
            ev = basis(np.array([0])).flatten()
            
        return S,P,ev,basis
        
    def solve(self, initial_tt, T, intervals = None, return_all = False,nswp = 40,qtt = False,verb = False,rounding = True):
        
       
        if intervals == None:
            pass
        else:
            x_tt = initial_tt
            dT = T / intervals
            Nt = self.N_max
            
            
                
            S,P,ev,basis = self.get_SP(dT,Nt)
            
            if qtt:
                nqtt = int(np.log2(Nt))
                S = ttm2qttm(tt.matrix(S))
                P = ttm2qttm(tt.matrix(P))
                I_tt = tt.eye(self.A_tt.n)
                B_tt = tt.kron(I_tt,tt.matrix(S)) - tt.kron(I_tt,P)@tt.kron(self.A_tt,ttm2qttm(tt.eye([Nt])))

            else: 
                nqtt = 1
                I_tt = tt.eye(self.A_tt.n)
                B_tt = tt.kron(I_tt,tt.matrix(S)) - tt.kron(I_tt,tt.matrix(P))@tt.kron(self.A_tt,tt.matrix(np.eye(Nt)))

            # print(dT,T,intervals)
            returns = []
            for i in range(intervals):
                # print(i)
                if qtt:
                    f_tt = tt.kron(x_tt,tt2qtt(tt.tensor(ev)))
                else: f_tt = tt.kron(x_tt,tt.tensor(  ev  ))
                # print(B_tt.n,f_tt.n)
                try:
                    # xs_tt = xs_tt.round(1e-10,5)
                    # tme = datetime.datetime.now()
                    xs_tt = tt.amen.amen_solve(B_tt, f_tt,  self.xs_tt, self.epsilon ,verb=1 if verb else 0,nswp = nswp,kickrank = 8,max_full_size=50,local_prec='n')
                    
                    # tme = datetime.datetime.now() - tme
                    # print(tme)
                    
                    self.xs_tt = xs_tt
                except:
                    # tme = datetime.datetime.now()
                    xs_tt = tt.amen.amen_solve(B_tt, f_tt,  f_tt, self.epsilon,verb=1 if verb else 0,nswp = nswp,kickrank = 8,max_full_size=50,local_prec='n')
                    # tme = datetime.datetime.now() - tme
                    # print(tme)
                    
                    self.xs_tt = xs_tt
                # print('SIZE',tt_size(xs_tt)/1e6)
                # print('PLMMM',tt.sum(xs_tt),xs_tt.r)
                if basis == None:
                    if return_all: returns.append(xs_tt)
                    x_tt = xs_tt[tuple([slice(None,None,None)]*len(self.A_tt.n)+[-1]*nqtt)]
                    x_tt = x_tt.round(self.epsilon/10)
                else:
                    
                    if return_all:
                        if qtt:
                            beval = basis(np.array([0])).flatten()
                            temp1 = xs_tt*tt.kron(tt.ones(self.A_tt.n),tt2qtt(tt.tensor(beval)))
                            for l in range(nqtt): temp1 = tt.sum(temp1,len(temp1.n)-1)
                            beval = basis(np.array([dT])).flatten()
                            temp2 = xs_tt*tt.kron(tt.ones(self.A_tt.n),tt2qtt(tt.tensor(beval)))
                            for l in range(nqtt): temp2 = tt.sum(temp2,len(temp2.n)-1)
                            returns.append(tt.kron(temp1,tt.tensor(np.array([1,0])))+tt.kron(temp2,tt.tensor(np.array([0,1]))))
                        else:
                            beval = basis(np.array([0])).flatten()
                            temp1 = xs_tt*tt.kron(tt.ones(self.A_tt.n),tt.tensor(beval))
                            temp1 = tt.sum(temp1,len(temp1.n)-1)
                            beval = basis(np.array([dT])).flatten()
                            temp2 = xs_tt*tt.kron(tt.ones(self.A_tt.n),tt.tensor(beval))
                            temp2 = tt.sum(temp2,len(temp2.n)-1)
                            returns.append(tt.kron(temp1,tt.tensor(np.array([1,0])))+tt.kron(temp2,tt.tensor(np.array([0,1]))))
 
                    beval = basis(np.array([dT])).flatten()
                    if qtt:
                        x_tt = xs_tt*tt.kron(tt.ones(self.A_tt.n),tt2qtt(tt.tensor(beval)))
                        for l in range(nqtt): x_tt = tt.sum(x_tt,len(x_tt.n)-1)
                        if rounding: x_tt = x_tt.round(self.epsilon/10)
                    else:
                        x_tt = tt.sum(xs_tt*tt.kron(tt.ones(self.A_tt.n),tt.tensor(beval)),len(xs_tt.n)-1)
                        if rounding: x_tt = x_tt.round(self.epsilon/10)
                # print('SIZE 2 ',tt_size(x_tt)/1e6)
            if not return_all: returns = x_tt 
            return returns
        