#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:37:52 2021

@author: yonnss
"""

import tt
import numpy as np


class IndependentGaussianObservation():
    
    def __init__(self,sigmas,N,observation_vector = None):
        self.sigmas = sigmas
        self.N = N
        self.observation_vector = len(N)*[1] if observation_vector==None else observation_vector
        
    def add_noise(self,sample):
        
        sample += np.random.normal(scale = self.sigmas, size=sample.shape)
        return sample
    
    def __call__(self,observation):
        
        lst = [ tt.tensor(np.exp(-0.5*(observation[i]-np.arange(self.N[i]))**2/self.sigmas[i]**2)/(self.sigmas[i]*np.sqrt(2*np.pi))) if self.observation_vector[i] else tt.ones([self.N[i]]) for i in range(len(self.N))]

        lst = [ tt.tensor(np.exp(-0.5*(observation[i]-np.arange(self.N[i]))**2/self.sigmas[i]**2)) if self.observation_vector[i] else tt.ones([self.N[i]]) for i in range(len(self.N))]
        tens = tt.mkron(lst)
        return tens
    
    
class IndependentLogNormalObservation():
    
    def __init__(self,sigmas,N,observation_vector = None):
        self.sigmas = sigmas
        self.N = N
        self.observation_vector = len(N)*[1] if observation_vector==None else observation_vector
        
    def add_noise(self,sample):
        
        lst = [ np.random.lognormal(np.log(sample[:,i]+1),self.sigmas[i]).reshape([-1,1]) for i in range(len(self.N)) ]
        
        sample = np.hstack(tuple(lst))
        
        return sample
    
    def __call__(self,observation):
        noise_model = lambda x,y,s : 1/(y*s*np.sqrt(2*np.pi)) * np.exp(-(np.log(y)-np.log(x+1))**2/(2*s**2))


        lst = [ tt.tensor(noise_model(np.arange(self.N[i]),observation[i],self.sigmas[i])) if self.observation_vector[i] else tt.ones([self.N[i]]) for i in range(len(self.N))]
        tens = tt.mkron(lst)
        return tens
    
    