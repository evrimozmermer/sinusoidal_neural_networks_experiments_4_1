# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:14:16 2020

@author: evrim
"""

import numpy as np
import torch as t
from scipy.signal import convolve

class SinDense:
    def __init__(self,layer_size,init_weights_range_in,init_weights_range_out,last_input_condition = False):
        self.layer_size = layer_size
        self.weights_in = False
        self.weights_out = False
        self.init_weights_range_in = init_weights_range_in
        self.init_weights_range_out = init_weights_range_out
        self.last_input_condition = last_input_condition
        self.pi = t.tensor(np.pi,dtype = t.float32)
    
    @staticmethod
    def random_uniform(lowest,highest,row,col):
        random_tensor = t.rand(int(row),int(col))*(highest-lowest)+lowest
        return random_tensor
        
    def dlayer_dlayer_func(self,layer,weights_in,weights_out):
        return weights_out*t.sin(self.pi*2*layer*weights_in)+self.pi*2*layer*weights_in*weights_out*t.cos(self.pi*4*layer*weights_in)
    
    def dlayer_dweights_func(self,in_,weights_in,weights_out):
        doutput_dweights_in = weights_out * (in_**2) * self.pi*4 * t.cos(self.pi*4*in_*weights_in)
        doutput_dweights_out = in_*t.sin(self.pi*4*in_*weights_in)
        return doutput_dweights_in,doutput_dweights_out

    def forward(self,input):
        if self.last_input_condition == True:
            self.last_input = input
        if type(self.weights_in) == bool:
            weight_ranges_in = [self.init_weights_range_in[0],self.init_weights_range_in[1]]
            weight_ranges_out = [self.init_weights_range_out[0],self.init_weights_range_out[1]]
            self.weights_in = self.random_uniform(weight_ranges_in[0], weight_ranges_in[1],self.layer_size,input.shape[0])
            self.weights_out = self.random_uniform(weight_ranges_out[0], weight_ranges_out[1],self.layer_size,input.shape[0])
        
        out = t.sum(self.weights_out*input*t.sin(self.pi*4*input*self.weights_in),axis = 1)
            
        #print(input.shape,self.weights_in.shape)
        return out
    
    def backprop(self,derror_dout,last_input = False,learning_rate = 0.05):
        last_input = self.last_input
        last_weights_in = self.weights_in
        last_weights_out = self.weights_out
        
        dout_dlayer = self.dlayer_dlayer_func(last_input,last_weights_in,last_weights_out)
        #print("dout_dlayer shape: ",dout_dlayer.shape)
        dout_dweights_in,dout_dweights_out = self.dlayer_dweights_func(last_input,last_weights_in,last_weights_out)
        #print("dout_dweights_in shape: ",dout_dweights_in.shape)
        #print("\n",derror_dout.shape,dout_dlayer.shape,"\n")
        derror_dlayer = t.mm(derror_dout,dout_dlayer)
        derror_dweights_in = derror_dout*dout_dweights_in.T
        derror_dweights_out = derror_dout*dout_dweights_out.T
        
        self.weights_in -= derror_dweights_in.T*learning_rate
        self.weights_out -= derror_dweights_out.T*learning_rate
        return derror_dlayer