#!/usr/bin/env python

import caffe
import numpy as np
from numpy import linalg as LA


class LqLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need 4 inputs to compute the ranking distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if (bottom[0].count != bottom[1].count) or (bottom[0].count != bottom[2].count) or (bottom[0].count != bottom[3].count) or (bottom[1].count != bottom[2].count) or (bottom[1].count != bottom[3].count) or (bottom[2].count != bottom[3].count):
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff1 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff2 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff3 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff4 = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        neg_dist = (bottom[0].data - bottom[1].data)
        c_dist = LA.norm(neg_dist)**2
        
        v_dist = (bottom[0].data - bottom[2].data)
        v_term = LA.norm(v_dist)**2
        
        o_dist = (bottom[0].data - bottom[3].data)
        o_term = LA.norm(o_dist)**2
        
        disc_term = v_term - o_term + 0.5 
        #print v_term, o_term, c_dist
        #raw_input()
        
        if (disc_term >= 0):
            self.diff1[...] = 2*neg_dist + 2*(v_dist) - 2*(o_dist)
        else: 
            self.diff1[...] = 2*neg_dist
        
        self.diff2[...] = -2*neg_dist
       
        if (disc_term >= 0):
            self.diff3[...] = -2*(v_dist)
        else: 
            self.diff3[...] = 0
        
        if (disc_term >= 0):
            self.diff4[...] = 2*(o_dist)
        else: 
            self.diff4[...] = 0
        
        d1 = np.sum(self.diff1**2) / bottom[0].num / 2.
        d2 = np.sum(self.diff2**2) / bottom[0].num / 2.
        d3 = np.sum(self.diff3**2) / bottom[0].num / 2.
        d4 = np.sum(self.diff4**2) / bottom[0].num / 2.
        
        top[0].data[...] = d1 + d2 + d3 + d4 

    def backward(self, top, propagate_down, bottom):
        for i in range(4):
            if not propagate_down[i]:
                continue
            if i == 0:
                grad_propag = self.diff1
            if i == 1:
                grad_propag = self.diff2
            if i == 2:
                grad_propag = self.diff3
            if i == 3:
                grad_propag = self.diff4
                
            bottom[i].diff[...] = grad_propag / bottom[i].num
