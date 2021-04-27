#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:30:16 2021

@author: ion
"""
import numpy as np
import matplotlib.pyplot 
import scipy
import scipy.interpolate
from scipy.interpolate import BSpline


def points_weights(a,b,nl):
    pts,ws = np.polynomial.legendre.leggauss(nl)
    pts = 0.5 * (b-a) * (pts+1) + a
    ws = (b-a) / 2  *ws
    return pts, ws



class LegendreBasis():
    def __init__(self, dim, domain = [-1,1]):
        self.dim = dim
        self.domain = domain
        self.basis = [ np.polynomial.legendre.Legendre.basis(i,domain) for i in range(dim) ]
        self.stiff = np.zeros((dim,dim))
        self.mass = np.zeros((dim,dim))
        self.ints = np.zeros(dim)
        
        pts,ws = points_weights(domain[0],domain[1], dim*2+4)
        Beval = self(pts)
        for i in range(dim):
            # pint = self.basis[i].integ()
            # self.ints[i] = pint(domain[1]) - pint(domain[0])
            self.ints[i] = np.sum(Beval[i,:]*ws)
            for j in range(dim):
                # pint = (self.basis[i] * self.basis[j]).integ()
                # self.mass[i,j] = pint(domain[1]) - pint(domain[0])
                self.mass[i,j] = np.sum(Beval[i,:]*Beval[j,:]*ws)
                
                # pint = (self.basis[i] * self.basis[j].deriv()).integ()
                # self.stiff[i,j] = pint(domain[1]) - pint(domain[0])
                
                
                
        
        
    def __call__(self,x,deriv = 0):
        result = []
        for b in self.basis:
            result.append(b.deriv(deriv)(x))
        return np.array(result)
    
    def get_integral(self):
        return self.ints
    
    def get_dimension(self):
        return self.dim
    
    def get_stiff(self):
        return self.stiff
    
    def get_mass(self):
        return self.mass
    
    def plot(self):
        x = np.linspace(self.domain[0],self.domain[1],self.dim*32)
        for b in self.basis:
            matplotlib.pyplot.plot(x,b(x))
    
    
    def interpolate(self,fun):
        
        pts,ws = np.polynomial.legendre.leggauss(self.dim*4)
        pts = 0.5 * (self.domain[1]-self.domain[0]) * (pts+1) + self.domain[0]
        ws = (self.domain[1]-self.domain[0]) / 2  *ws
        
        vals = fun(pts)*ws
        
        b = np.sum(self(pts) * vals,1).reshape([-1,1])
       
        return np.linalg.solve(self.mass,b).flatten()
       
    def integration_points(self,mult = 3):
        
            
        p, w = points_weights(self.domain[0],self.domain[1], self.dim*mult)
        
        return p, w
    
    
class ChebyBasis():
    def __init__(self, dim, domain = [-1,1]):
        self.dim = dim
        self.domain = domain
        self.basis = [ np.polynomial.legendre.Legendre.basis(i,domain) for i in range(dim) ]
        self.stiff = np.zeros((dim,dim))
        self.mass = np.zeros((dim,dim))
        self.ints = np.zeros(dim)
        
        for i in range(dim):
            pint = self.basis[i].integ()
            self.ints[i] = pint(domain[1]) - pint(domain[0])
            for j in range(dim):
                pint = (self.basis[i] * self.basis[j]).integ()
                self.mass[i,j] = pint(domain[1]) - pint(domain[0])
                pint = (self.basis[i] * self.basis[j].deriv()).integ()
                self.stiff[i,j] = pint(domain[1]) - pint(domain[0])
        
    def __call__(self,x,deriv = 0):
        result = []
        for b in self.basis:
            result.append(b.deriv(deriv)(x))
        return np.array(result)
    
    def get_integral(self):
        return self.ints
    
    def get_dimension(self):
        return self.dim
    
    def get_stiff(self):
        return self.stiff
    
    def get_mass(self):
        return self.mass
    
    def plot(self):
        x = np.linspace(self.domain[0],self.domain[1],self.dim*32)
        for b in self.basis:
            matplotlib.pyplot.plot(x,b(x))
            
    def interpolate(self,fun):
        
        pts,ws = np.polynomial.legendre.leggauss(self.dim*4)
        pts = 0.5 * (self.domain[1]-self.domain[0]) * (pts+1) + self.domain[0]
        ws = (self.domain[1]-self.domain[0]) / 2  *ws
        
        vals = fun(pts)*ws
        
        b = np.sum(self(pts) * vals,1).reshape([-1,1])
       
        return np.linalg.solve(self.mass,b).flatten()

def lagrange_basis(x,k,n):
    
    y= np.poly1d([1.0])
    const = 1
    for j in range ( n ):
        if k!=j:
            y*=np.poly1d([1,-x[j]]) 
            const *= ( x[k]-x[j] )
    return y/const

class LagrangeBasis():
    def __init__(self, dim, domain = [-1,1]):
        self.dim = dim
        self.domain = domain
        pts,_ = np.polynomial.chebyshev.chebgauss(dim)
        self.pts = 0.5 * (domain[1]-domain[0]) * (pts+1) + domain[0]
        # self.pts = np.linspace(domain[0],domain[1],dim)
        # self.basis = [ scipy.interpolate.lagrange(self.pts,np.eye(dim)[:,i]) for i in range(dim) ]
        self.basis = [ lagrange_basis(self.pts,i,dim) for i in range(dim) ]
        self.stiff = np.zeros((dim,dim))
        self.mass = np.zeros((dim,dim))
        self.ints = np.zeros(dim)
        
        for i in range(dim):
            pint = self.basis[i].integ()
            self.ints[i] = pint(domain[1]) - pint(domain[0])
            for j in range(dim):
                pint = (self.basis[i] * self.basis[j]).integ()
                self.mass[i,j] = pint(domain[1]) - pint(domain[0])
                pint = (self.basis[i] * self.basis[j].deriv()).integ()
                self.stiff[i,j] = pint(domain[1]) - pint(domain[0])
        
    def __call__(self,x,deriv = 0):
        result = []
        for b in self.basis:
            result.append(b.deriv(deriv)(x))
        return np.array(result)
    
    def get_dimension(self):
        return self.dim
    
    def get_stiff(self):
        return self.stiff
    
    def get_integral(self):
        return self.ints.flatten()
    
    def get_mass(self):
        return self.mass
    
    def plot(self):
        x = np.linspace(self.domain[0],self.domain[1],self.dim*32)
        for b in self.basis:
            matplotlib.pyplot.plot(x,b(x))
            
    def integration_points(self,mult = 2):
        return points_weights(self.domain[0], self.domain[1], self.deg*mult)
        
    def interpolate(self,function):
        return function(self.pts)
        
class BSplineBasis:
    
    def __init__(self, dim, domain = [-1,1],deg = 1):
        
        self.dim = dim
        self.deg = deg
        self.domain = domain
        knots = np.linspace(domain[0],domain[1],dim+1-deg)
        self.N=knots.size+deg-1
        self.deg=deg
        self.knots=np.hstack( ( np.ones(deg)*knots[0] , knots , np.ones(deg)*knots[-1] ) )
        self.spl = []
        self.dspl = []
        for i in range(self.N):
            c=np.zeros(self.N)
            c[i]=1
            self.spl.append(BSpline(self.knots,c,self.deg))
            self.dspl.append(scipy.interpolate.splder( BSpline(self.knots,c,self.deg) ))
        
        self.compact_support_bsp = np.zeros((self.N,2))
        for i in range(self.N):
            self.compact_support_bsp[i,0] = self.knots[i]
            self.compact_support_bsp[i,1] = self.knots[i+self.deg+1]
            
        int_bsp_bsp = np.zeros((self.N,self.N))
        int_bsp = np.zeros((self.N,1))
        # int_bsp_v = np.zeros((self.Nz,1))
        
        Pts, Ws =np.polynomial.legendre.leggauss(20)
        for i in range(self.N):
            a=self.compact_support_bsp[i,0]
            b=self.compact_support_bsp[i,1]

            for k in range(self.knots.size-1):
                if self.knots[k]>=a and self.knots[k+1]<=b:
                    pts = self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k])
                    ws = Ws*(self.knots[k+1]-self.knots[k])/2
                    int_bsp[i,0] += np.sum( self.__call__(pts,i) * ws )
                    
            for j in range(i,self.N):
                a=self.compact_support_bsp[j,0]
                b=self.compact_support_bsp[i,1]
                if b>a:
                    for k in range(self.knots.size-1):
                        if self.knots[k]>=a and self.knots[k+1]<=b:
                            pts = self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k])
                            ws = Ws*(self.knots[k+1]-self.knots[k])/2
                            int_bsp_bsp[i,j] += np.sum(  self.__call__(pts,i) *self.__call__(pts,j) * ws )
                            # int_bspp[i,j] += np.sum( self.bspp(pts)[i,:]* self.bspp(pts)[j,:]*ws )
                    if i!=j:
                        int_bsp_bsp[j,i] = int_bsp_bsp[i,j]
                        # int_bspp[j,i] = int_bspp[i,j]
                    
        
        self.int_bsp_bsp = int_bsp_bsp
        # self.int_bspp_bspp = int_bspp
        self.int_bsp = int_bsp
        
    
    def __call__(self,x,i=None,derivative=False):
        if i==None:
            if derivative:
                ret = np.array([self.dspl[i](x) for i in range(self.N)])
                return ret
            else:
                ret = np.array([self.spl[i](x) for i in range(self.N)])
                return ret
        else:
            if derivative:
                 return self.dspl[i](x)
            else:
                return self.spl[i](x)
    def greville(self):
        return np.array([np.sum(self.knots[i+1:i+self.deg+1]) for i in range(self.N)])/(self.deg)
        
    def eval_all(self,c,x):
        c=np.hstack((c,np.zeros(self.deg-2)))
        return BSpline(self.knots,c,self.deg)(x)
    
    def get_dimension(self):
        return self.dim
    
    def get_stiff(self):
        return None
    
    def get_integral(self):
        return self.int_bsp.flatten()
    
    def get_mass(self):
        return self.int_bsp_bsp
    
    def derivative(self):
        bd = scipy.interpolate.splder(BSpline(self.knots,np.zeros(self.N+self.deg-1)+1,self.deg))
        return BSplineBasis(np.unique(bd.t), bd.k)
    
    def interpolate(self,fun):
        
        xg = self.greville().flatten()
        yg = fun(xg)
        Gm = self(xg)
        
        dofs = np.linalg.solve(Gm,yg.reshape([-1,1])).flatten()
        
        return dofs
    
    def integration_points(self,mult = 2):
        pts = []
        ws = []
        
        ku = np.unique(self.knots)
        for i in range(ku.size-1):
            
            p, w = points_weights(ku[i], ku[i+1], self.deg*mult)
            
            pts += list(p)
            ws += list(w)
            
        return np.array(pts), np.array(ws)
            
    