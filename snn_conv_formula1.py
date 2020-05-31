# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:14:16 2020

@author: evrim
"""

import numpy as np
import torch as t
from scipy.signal import convolve

class SinDense:
    def __init__(self,layer_size,init_weights_range,last_input_condition = False):
        self.layer_size = layer_size
        self.weights_in = False
        self.init_weights_range = init_weights_range
        self.last_input_condition = last_input_condition
        self.pi = t.tensor(np.pi,dtype = t.float32)
    
    @staticmethod
    def random_uniform(lowest,highest,row,col):
        random_tensor = t.rand(int(row),int(col))*(highest-lowest)+lowest
        return random_tensor
        
    def dlayer_dlayer_func(self,layer,weights_in):
        return weights_in * self.pi*2 * t.cos(self.pi*2*layer*weights_in)
    
    def dlayer_dweights_func(self,in_,weights_in):
        doutput_dweights_in = in_ * self.pi*2 * t.cos(self.pi*2*in_*weights_in)
        return doutput_dweights_in

    def forward(self,input):
        if self.last_input_condition == True:
            self.last_input = input
        if type(self.weights_in) == bool:
            weight_ranges = [self.init_weights_range[0],self.init_weights_range[1]] 
            self.weights_in = self.random_uniform(weight_ranges[0], weight_ranges[1],self.layer_size,input.shape[0])
        
        out = t.sum(0.1*t.sin(self.pi*2*input*self.weights_in),axis = 1)
            
        #print(input.shape,self.weights_in.shape)
        return out
    
    def backprop(self,derror_dout,last_input = False,learning_rate = 0.05):
        last_input = self.last_input
        last_weights_in = self.weights_in
        
        dout_dlayer = self.dlayer_dlayer_func(last_input,last_weights_in)
        dout_dweights_in = self.dlayer_dweights_func(last_input,last_weights_in)
        
        #print(derror_dout.shape,dout_dlayer.shape)
        #print("__",derror_dout.shape,dout_dlayer.shape)
        derror_dlayer = t.mm(derror_dout,dout_dlayer)
        derror_dweights_in = derror_dout*dout_dweights_in.T
        #print(derror_dweights_in.shape,derror_dweights_in.T.shape,self.weights_in.shape)
        self.weights_in -= derror_dweights_in.T*learning_rate
        return derror_dlayer