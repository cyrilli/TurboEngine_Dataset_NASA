#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:38:22 2017

@author: cyril
"""
import numpy as np

def load_data(data_path, data_name):
    data = np.load(data_path + data_name)
    return data
    
def gen_batch(raw_data, batch_size, shuffle=True):
    '''
    during each iter, randomly select a mini batch (batch_size, num_steps) from raw_data
    in the next iter, the data that has already been trained on won't be chosen again
    Input:  raw_data  numpy array (data_length, num_steps, num_features + 1), label in the last column
            batch_size  integer
    Output: (x, y)
            x, numpy array (batch_size, num_steps, num_features)
            y, numpy array (batch_size, num_steps)
    '''  
    data_length = raw_data.shape[0]
    if shuffle:
        indices = np.arange(data_length)
        np.random.shuffle(indices)
    for start_idx in range(0, data_length - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        batch_raw_data = raw_data[excerpt]   # (batch_size, num_steps, num_features+1)
        yield batch_raw_data[:,:,0:-1], batch_raw_data[:,:,-1]
        
def gen_epochs(max_epoch, batch_size, data):
    for i in range(max_epoch):
        yield gen_batch(data, batch_size, shuffle = True) # the input data is padded