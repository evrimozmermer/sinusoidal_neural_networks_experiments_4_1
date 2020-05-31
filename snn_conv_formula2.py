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
        self.weights_out = False
        self.init_weights_range = init_weights_range
        self.last_input_condition = last_input_condition
        self.pi = t.tensor(np.pi,dtype = t.float32)
    
    @staticmethod
    def random_uniform(lowest,highest,row,col):
        random_tensor = t.rand(int(row),int(col))*(highest-lowest)+lowest
        return random_tensor
        
    def dlayer_dlayer_func(self,layer,weights_in,weights_out):
        return weights_out * weights_in * self.pi*2 * t.cos(self.pi*2*layer*weights_in)
    
    def dlayer_dweights_func(self,in_,weights_in,weights_out):
        doutput_dweights_in = weights_out * in_ * self.pi*2 * t.cos(self.pi*2*in_*weights_in)
        doutput_dweights_out = t.sin(self.pi*2*in_*weights_in)
        return doutput_dweights_in,doutput_dweights_out

    def forward(self,input):
        if self.last_input_condition == True:
            self.last_input = input
        if type(self.weights_in) == bool:
            weight_ranges = [self.init_weights_range[0],self.init_weights_range[1]] 
            self.weights_in = self.random_uniform(weight_ranges[0], weight_ranges[1],self.layer_size,input.shape[0])
            self.weights_out = self.random_uniform(weight_ranges[0], weight_ranges[1],self.layer_size,input.shape[0])
        
        out = t.sum(self.weights_out*t.sin(self.pi*2*input*self.weights_in),axis = 1)
            
        #print(input.shape,self.weights_in.shape)
        return out
    
    def backprop(self,derror_dout,last_input = False,learning_rate = 0.05):
        last_input = self.last_input
        last_weights_in = self.weights_in
        last_weights_out = self.weights_out
        
        dout_dlayer = self.dlayer_dlayer_func(last_input,last_weights_in,last_weights_out)
        dout_dweights_in,dout_dweights_out = self.dlayer_dweights_func(last_input,last_weights_in,last_weights_out)
        
        #print("__",derror_dout.shape,dout_dlayer.shape)
        derror_dlayer = t.mm(derror_dout,dout_dlayer)
        derror_dweights_in = derror_dout*dout_dweights_in.T
        derror_dweights_out = derror_dout*dout_dweights_out.T
        
        self.weights_in -= derror_dweights_in.T*learning_rate
        self.weights_out -= derror_dweights_out.T*learning_rate
        return derror_dlayer

class SinConv(SinDense): #SinDense is parent class
    def __init__(self, num_filters,filter_shape = [3,3],init_weights_range = [-0.3,0.3],strides = False):
        self.num_filters = num_filters
        self.filter_shape = filter_shape
        self.filters = False
        self.SinDense = SinDense
        self.init_weights_range = init_weights_range
        self.strides = strides
    
    @staticmethod
    def random_uniform(lowest,highest,num_filters,chan,row,col):
        random_tensor = t.rand(num_filters,chan,row,col)*(highest-lowest)+lowest
        return random_tensor
    
    def defineFilterClasses(self,Row,Column,Channel):
        #filterClassShape = shape of output matrix
        filterClasses = []
        for chan in range(Channel):
            filterClassesElm = []
            sd = SinDense(1,self.init_weights_range)
            for row in range(Row):
                filterClassesElm.append([])
                for col in range(Column):
                    filterClassesElm[-1].append(sd) #Append SinDense instances to list for each filter.
            filterClasses.append(filterClassesElm)
        return filterClasses #This is actually not filters but list of SinDense instances
                            
    def forward(self,img):
        self.last_input = img
        #print(img.shape)
        _,chani,r_img,c_img = img.shape
        if type(self.filters) == bool:
            weight_ranges = [self.init_weights_range[0],self.init_weights_range[1]]
            #print((chani,self.filter_shape[0],self.filter_shape[1],self.num_filters))
            self.filters = self.random_uniform(weight_ranges[0], weight_ranges[1],self.num_filters,chani,self.filter_shape[0],self.filter_shape[1])
        
        #print(img.unsqueeze(0).shape,self.filters.shape)
        if not self.strides:
            pad = int((self.filter_shape[0]-1)/2)
        else:
            pad = 0
        processed = t.nn.functional.conv2d(img, self.filters, padding = pad, stride = self.strides)
        
        #print("___",type(processed))
        return processed

    def backprop(self,derror_dout,learning_rate = 0.05):
        last_input = self.last_input
        d_L_d_input = np.zeros(last_input.shape)
        r_img,c_img,chani = last_input.shape
        N,r_filt,c_filt,chanf = self.num_filters,self.filter_shape[0],self.filter_shape[1],chani
        for n in range(N):
            for row in range(r_img-(r_filt-1)):
                for col in range(c_img-(c_filt-1)):
                    kernel = last_input[row:row+r_filt,col:col+c_filt]
                    #print(derror_dout[row,col,n],kernel.flatten())
                    #print("kernel___", type(kernel))
                    d_L_d_input[row:row+r_filt,col:col+c_filt] = self.filters[row,col,n].backprop(derror_dout[row,col,n],kernel.flatten(),learning_rate = learning_rate).reshape((kernel.shape[0],kernel.shape[1],chanf))    
        return d_L_d_input

class SinDeConv:
    def __init__(self, num_filters,filter_shape = [3,3],init_weights_range = [-0.3,0.3]):
        self.num_filters = num_filters
        self.filter_shape = filter_shape
        self.filters = False
        self.SinDense = SinDense
        self.init_weights_range = init_weights_range
    
    def defineFilterClasses(self,Row,Column,Channel,deconv_filter_shape):
        #filterClassShape = shape of output matrix
        deconv_filter_size = deconv_filter_shape[0]*deconv_filter_shape[1]
        filterClasses = []
        for chan in range(Channel):
            filterClassesElm = []
            sd = SinDense(1,self.init_weights_range)
            for row in range(Row):
                filterClassesElm.append([deconv_filter_size])
                for col in range(Column):
                    filterClassesElm[-1].append(sd) #Append SinDense instances to list for each filter.
            filterClasses.append(filterClassesElm)
        return filterClasses #This is actually not filters but list of SinDense instances
                            
    def forward(self,img):
        self.last_input = img
        r_img,c_img,chani = img.shape
        N,r_filt,c_filt,chanf = self.num_filters,self.filter_shape[0],self.filter_shape[1],chani
        if type(self.filters) == bool:
            self.filters = self.defineFilterClasses(r_img-(r_filt-1),c_img-(c_filt-1),self.num_filters)
            self.filters = np.moveaxis(np.array(self.filters),0,2)
            
        processed = np.zeros((r_img-(r_filt-1),c_img-(c_filt-1),N))
        for n in range(processed[2]):
            for row in range(processed[0]):
                for col in range(processed[1]):
                    kernel = img[row:row+r_filt,col:col+c_filt]
                    #print(processed[row,col,n],self.filters[row,col,n].forward(kernel.flatten()))
                    operation_result = self.filters[row,col,n].forward(kernel.flatten())
                    try:
                        processed[row,col,n] = operation_result
                    except:
                        processed[row,col,n] = operation_result[0] #perform forward operation in SinDense instances
        return np.asarray(processed)

    def backprop(self,derror_dout,learning_rate = 0.05):
        last_input = self.last_input
        d_L_d_input = np.zeros(last_input.shape)
        r_img,c_img,chani = last_input.shape
        N,r_filt,c_filt,chanf = self.num_filters,self.filter_shape[0],self.filter_shape[1],chani
        for n in range(N):
            for row in range(r_img-(r_filt-1)):
                for col in range(c_img-(c_filt-1)):
                    kernel = last_input[row:row+r_filt,col:col+c_filt]
                    #print(derror_dout[row,col,n],kernel.flatten())
                    #print("kernel___", type(kernel))
                    d_L_d_input[row:row+r_filt,col:col+c_filt] = self.filters[row,col,n].backprop(derror_dout[row,col,n],kernel.flatten(),learning_rate = learning_rate).reshape((kernel.shape[0],kernel.shape[1],chanf))    
        return d_L_d_input

class MeanPool:
    def __init__(self):
        
        pass
        
    def forward(self,img):
        pooled = t.nn.functional.avg_pool2d(img, (2,2), stride=2)
        return pooled
        
class MedianPool:
    def __init__(self):
        pass
        
    def forward(self,img):
        dummy,row,col,chani = img.shape
        #forBackprop = np.zeros(img.shape)
        
        if row%2 == 0:
            newRow = int(row/2)
        elif row%2 == 1:
            newRow = int((row-1)/2)
        if col%2 == 0:
            newCol = int(col/2)
        elif col%2 == 1:
            newCol = int((col-1)/2)
            
        if row%2 == 0:
            row = int(row)
        elif row%2 == 1:
            row = int(row-1)
        if col%2 == 0:
            col = int(col)
        elif col%2 == 1:
            col = int(col-1)
            
        forForward = np.zeros((newRow,newCol,chani))
        row_cnt = 0
        #print("__",img.shape)
        for r in range(0,row,2):
            col_cnt = 0
            for c in range(0,col,2):
                for chan in range(0,chani,1):
                    kernel = img[0,r:r+2,c:c+2,chan]
                    medval = np.median(kernel)
                    #forBackprop[r:r+2,c:c+2,chan][np.where(kernel==maxval)] = 1
                    forForward[row_cnt,col_cnt,chan] = medval
                col_cnt += 1
            row_cnt += 1
                    
            #print("chan finished",r)
        #print("all chan finished")
        #forForward = np.asarray(forForward)
        #self.forBackprop = forBackprop        
        return forForward
    
    def backprop(self,error):
        newError = error.repeat(2, axis=0).repeat(2, axis=1)
        #forBackprop = self.forBackprop
        #newError = np.multiply(newError[0:forBackprop.shape[0],
        #                                0:forBackprop.shape[1]],forBackprop)
        return newError

class MaxPool:
    def __init__(self):
        pass
        
    def forward(self,img):
        dummy,row,col,chani = img.shape
        forBackprop = np.zeros(img.shape)
        
        if row%2 == 0:
            newRow = int(row/2)
        elif row%2 == 1:
            newRow = int((row-1)/2)
        if col%2 == 0:
            newCol = int(col/2)
        elif col%2 == 1:
            newCol = int((col-1)/2)
            
        if row%2 == 0:
            row = int(row)
        elif row%2 == 1:
            row = int(row-1)
        if col%2 == 0:
            col = int(col)
        elif col%2 == 1:
            col = int(col-1)
            
        forForward = np.zeros((newRow,newCol,chani))
        row_cnt = 0
        for r in range(0,row,2):
            col_cnt = 0
            for c in range(0,col,2):
                for chan in range(0,chani,1):
                    kernel = img[0,r:r+2,c:c+2,chan]
                    maxval = np.max(kernel)
                    forBackprop[0,r:r+2,c:c+2,chan][np.where(kernel==maxval)] = 1
                    forForward[row_cnt,col_cnt,chan] = maxval
                col_cnt += 1
            row_cnt += 1
                    
            #print("chan finished",r)
        #print("all chan finished")
        #forForward = np.asarray(forForward)
        self.forBackprop = forBackprop        
        return forForward
    
    def backprop(self,error):
        newError = error.repeat(2, axis=0).repeat(2, axis=1)
        forBackprop = self.forBackprop
        newError = np.multiply(newError[0:forBackprop.shape[0],
                                        0:forBackprop.shape[1]],forBackprop)
        return newError
        
class Activations:
    def __init__(self):
        pass
    
    @staticmethod
    def sine_activation(output):
        output_act = np.sin((np.pi/2)*output)
        return output_act
    
    @staticmethod
    def sine_loss_1(output_act,error):
        derivative = (np.pi/2)*np.cos((np.pi/2)*output_act)
        return error*derivative
        
    @staticmethod
    def error_function_for_sin_single(output,y):
        to_search_max = np.sin(np.linspace(output-1,output+1,10000)*np.pi/2)
        index = np.argmin(np.abs(to_search_max-y))
        to_be = np.linspace(output-1,output+1,10000)[index]
        error = to_be-output
        #print("to be:",to_be,"as is",output,"error",error/10)
        return error

    def sine_loss_2(self,output,y):
        derror_doutput = []
        for cnt in range(y.shape[0]):
            derror_doutput.append(self.error_function_for_sin_single(output[cnt],y[cnt]))
        derror_doutput = np.array(derror_doutput)
        #print("____________")
        return derror_doutput
    
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    @staticmethod
    def softmax_grad(softmax):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    
class Losses:
    @staticmethod
    def logloss(true_label, predicted, eps=1e-15):
        p = np.clip(predicted[0], eps, 1 - eps)
        loss = []
        cnt = 0
        #print(p)
        for elm in true_label:
            if elm == 1:
                loss.append(-np.log(p[cnt]))
            else:
                loss.append(-np.log(1 - p[cnt]))
            cnt += 1
        return np.array(loss)