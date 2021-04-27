#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:14:12 2021

@author: ion
"""

import tt
import numpy as np
from tt_aux import *
from basis import *

def GammaPDF(alpha, beta, basis, lb,ub):
    pdf = pdfTT([basis],[[lb,ub]])
    # print(basis.interpolate(lambda x : x**(alpha-1) * np.exp(-beta*x)))
    pdf.update(tt.tensor(basis.interpolate(lambda x : x**(alpha-1) * np.exp(-beta*x))))
    pdf.normalize()        
    return pdf


def UniformPDF(basis, domain = [0,1]):
    if isinstance(basis,list):
        pdf = pdfTT(basis,domain)
        # pdf.update(tt.mkron([tt.tensor(b.interpolate(lambda x : x*0+1)) for b in basis]))
        pdf.update(tt.mkron([tt.tensor(np.ones(b.get_dimension())) for b in basis]))
    else:
        pdf = pdfTT([basis],[domain])
        pdf.update(tt.tensor(basis.interpolate(lambda x : x*0+1)))
        pdf.update(tt.ones([basis.get_dimension()]))
    pdf.normalize()  
    return pdf

def SingularPMF(N,I):
    
    return tt.tensor().from_list([np.eye(N[i])[I[i],:].reshape([1,N[i],1]) for i in range(len(N))])
    
    
class pdfTT():
    def __init__(self,basis,compact_support):
        
        self.d = len(basis)
        self.N = [b.get_dimension() for b in basis]
        self.basis = basis
        self.compact_support = compact_support
        self.tt = tt.ones(self.N)
        
    def update(self,tensor):
        self.tt = tensor
        
    def normalize(self,probability_mass=True):
        if probability_mass:
            int_tt = tt.mkron([tt.tensor(b.get_integral()) for b in self.basis ])
            Z = tt.dot(self.tt,int_tt)
            self.tt = self.tt*(1/Z)
            
    
    def expected_value(self):
        '''
        Compute the expected value of the pdf.

        Returns
        -------
        E : np array
            the expected value.

        '''
        E = np.zeros((self.d))
        for i in range(self.d):
            pts, ws = self.basis[i].integration_points(4)
            temp = np.einsum('ij,j->i',self.basis[i](pts) * pts ,ws)

            
            temp = tt.tensor(temp)
            E[i] = tt.dot(tt.mkron([tt.tensor(self.basis[k].get_integral()) if k!=i else temp for k in range(self.d)]),self.tt)
            
        return E
    
    def covariance_matrix(self):
        '''
        Compute the expected value of the pdf.

        Returns
        -------
        E : np array
            the expected value.

        '''
        C = np.zeros((self.d,self.d))
        E = self.expected_value()
        
        Pts = [b.integration_points(4)[0] for b in self.basis]
        Ws = [b.integration_points(4)[1] for b in self.basis]
        Bs = [self.basis[k](Pts[k]) for k in range(self.d)]
        w_tt = tt.mkron([tt.tensor(w) for w in Ws])
        
        for i in range(self.d):
            
            for j in range(i,self.d):
                
                Iop = tt.matrix().from_list([(Bs[k]*(Pts[k] if k==i else 1)*(Pts[k] if k==j else 1)).reshape([1,Bs[k].shape[0],Bs[k].shape[1],1]) for k in range(self.d)])
                
                C[i,j] = tt.dot( tt.matvec(Iop.T,self.tt) , w_tt ) - E[i]*E[j]
            
        return C
    
    def marginal(self,mask):
        
        ints = [tt.tensor(self.basis[k].get_integral()) if not mask[k] else tt.ones([self.basis[k].get_dimension()]) for k in range(self.d)]
        
        basis_new = []
        compact_new = []
        k = 0
        tt_new = self.tt.copy()
        for i in range(self.d):
            if mask[i]:
                basis_new.append(self.basis[i])
                compact_new.append(self.compact_support[i])
            else:
                tt_new = tt.sum(tt_new,i-k)
                k+=1
        pdf_new = pdfTT(basis_new, compact_new)
        # print(basis_new,compact_new,tt_new)
        pdf_new.tt = tt_new
        pdf_new.normalize()
        return pdf_new
    

    
    def round(self,eps=1e-12,rmax=9999):
        self.tt = self.tt.round(epsilon,rank)
    
    def __call__(self,x):
        
        beval = [tt.tensor(self.basis[i](x[i]).flatten()) for i in range(self.d)]
        
        return tt.dot(self.tt,tt.mkron(beval))
        
        
    def __pow__(self,other):
        
        basis_new = self.basis + other.basis
        compact_new = self.compact_support + other.compact_support
   
        pdf = pdfTT(basis_new,compact_new)
        pdf.tt = tt.kron(self.tt, other.tt)
        pdf.normalize()
        return pdf
                
def get_mass(basis):
    lst = [b.get_mass().reshape([1,b.get_dimension(),b.get_dimension(),1]) for b in basis]
    lst_inv = [np.linalg.inv(b.get_mass()).reshape([1,b.get_dimension(),b.get_dimension(),1]) for b in basis]
    return tt.matrix().from_list(lst), tt.matrix().from_list(lst_inv)

def get_stiff(Att_extended,N,pts_list,ws_list,basis):
    Np = len(basis)
    
    lst_cores = Att_extended.to_list(Att_extended)
    
    for i in range(len(N),len(N)+Np):
        
        coreA = lst_cores[i]
        coreAA = lst_cores[i]
        P = basis[i-len(N)](pts_list[i-len(N)])
        # print(i, i-len(N),P.shape,ws_list[i-len(N)].shape,np.sum(ws_list[i-len(N)]),basis[i-len(N)].domain)
        coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[i-len(N)]))
        coreA = np.einsum('abcd,nb->ancd',coreA,P)
        coreA = np.einsum('ancd,lc->anld',coreA,P)
        
        core_new = np.zeros((coreAA.shape[0],coreAA.shape[1],coreAA.shape[3]))
        for p in range(basis[i-len(N)].get_dimension()):
            core_new[:,p,:] = coreAA[:,p,p,:]
            
        core_new = np.einsum('apb,p,mp,lp->amlb',core_new,ws_list[i-len(N)],P,P)
            
        # print(np.linalg.norm(core_new-coreA)/np.linalg.norm(core_new))
        # coreA = np.einsum('anld,nl->anld',coreA,P)
        
        # coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[len(N)-i]))
        # print(coreAA[-1,:,:,-1])
        lst_cores[i] = coreA

    Aext = tt.matrix().from_list(lst_cores)
    return Aext
 
def interpolate_cme_tt(Att_extended,N,pts_list,ws_list,basis):
    Np = len(basis)
    
    lst_cores = Att_extended.to_list(Att_extended)
    
    for i in range(len(N),len(N)+Np):
        
        coreA = lst_cores[i]
        
        
        core_new = np.zeros((coreA.shape[0],coreAA.shape[1],coreAA.shape[3]))
        for p in range(basis[len(N)-i].get_dimension()):
            core_new[:,p,:] = coreAA[:,p,p,:]
            
        core_new = np.einsum('apb,p,mp->amb',core_new,ws_list[len(N)-i],P)
            
        print(np.linalg.norm(core_new-coreA)/np.linalg.norm(core_new))
        # coreA = np.einsum('anld,nl->anld',coreA,P)
        
        # coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[len(N)-i]))
        lst_cores[i] = core_new

    Aext = tt.matrix().from_list(lst_cores)
    return Aext



def extend_cme(Alist, pts_rates):
    n = len(Alist)
    
    for i,pts in enumerate(pts_rates):
        if pts.size!=1:
            for k in range(n):
                Alist[k] = tt.kron(Alist[k],tt.eye([pts.size]) if k!=i else tt.matrix(np.diag(pts)))
        else:
            Alist[i] = Alist[i]*pts[0]
            
    Att = Alist[0]*0
    for A in Alist: Att += A
    
    return Att
    

# baza = LagrangeBasis(128,[0,20])
# baza = LegendreBasis(128,[0,20])
# pr = GammaPDF(7.5,1,baza,0,20)
# pr2 = pr ** GammaPDF(4.5,3,baza,0,20)

# print(pr.expected_value())
# print(pr2.expected_value())
# print(pr2.covariance_matrix())

# import matplotlib.pyplot as plt
# from scipy.stats import gamma

# x = np.linspace(0,20,1000)
# B = baza(x)

# plt.figure()
# plt.plot(x,np.einsum('ij,i->j',B,pr.tt.full()))
# plt.plot(x,gamma.pdf(x,7.5))
