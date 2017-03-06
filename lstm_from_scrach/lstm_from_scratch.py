#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:25:27 2017

@author: cyril
"""
import tensorflow as tf
import tensorlayer as tl
import data_utilities as utils

# load the training data that has already been padded, shape: (data_length, num_steps, num_features + 1) 1 stands for the label
data_path = '/home/cyril/TurboEngine_Dataset_NASA/'
train_data = utils.load_data(data_path, 'padded_train_normalized.npy')  # (709, 543, 25)

'''
Building up the model using dynamic_rnn
Input x: [batch_size, n_steps(max), n_features]
'''
##########     Setting Hyper-parameters    ##########
num_units = 100  # number of hidden units
batch_size = 80  # the number of sequences in a batch
max_epoch = 100  
num_layers = 4  # the number of LSTM layers
learning_rate = 1
num_steps = 543  # max_time
dropout_rate = 0.8
# fixed param
num_features = 24  # for CMAPPS dataset, this includes settings 1,2,3 and s1,2,3,....21

############      Model Configuration      ###########
x = tf.placeholder(dtype=tf.float64, shape=[batch_size, num_steps, num_features], name="x")
y = tf.placeholder(dtype=tf.float64, shape=[batch_size, num_steps], name="y")

#seq_length_of_the_batch = tl.layers.retrieve_seq_length_op2(x) # a Tensor [1, batch_size]
seq_length_of_the_batch = tl.layers.retrieve_seq_length_op(x if isinstance(x, tf.Tensor) else tf.stack(x))
seq_length_of_the_batch = tf.cast(seq_length_of_the_batch, tf.float64)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=dropout_rate)
cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * num_layers, state_is_tuple=True)
init_state = cell.zero_state(batch_size, tf.float32)  # set the initial state of LSTM to be zero
rnn_outputs, last_states = tf.nn.dynamic_rnn(
                                            cell=cell,
                                            dtype=tf.float64,
                                            sequence_length=seq_length_of_the_batch,
                                            inputs=x)
# outputs: [batch_size, max_time, num_units]
# last_states: (c, h)

'''
CALCULATING SEQUENCE LOSS ON PADDED EXAMPLES:
    Create a weight matrix that “masks out” the losses at padded positions
    Use tf.sign(tf.to_float(y)) to create a mask
    Side Note: we can only do this when we don't have “0-class”
'''
tf.sign(tf.to_float(y))

rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, num_units]) # (batch_size * max_time, num_units)
y_flat = tf.reshape(y, [-1])   # or maybe [-1,1]???

with tf.variable_scope('regression'):
    regression_w = tf.get_variable('regression_w', [num_units, 1], dtype=tf.float64)
    regression_b = tf.get_variable('regression_b', [1], initializer=tf.constant_initializer(0.0),dtype=tf.float64)

prediction = tf.matmul(rnn_outputs_flat, regression_w) + regression_b  #(batch_size * num_steps, 1) 
prediction = tf.reshape(prediction, [-1])
losses = tf.pow(prediction-y_flat,2)  # (batch_size * num_steps, 1)

# Mask the losses
mask = tf.sign(tf.cast(y_flat, tf.float64))
masked_losses = mask * losses

# Bring back to (batch_size, num_steps) shape
masked_losses = tf.reshape(masked_losses,  tf.shape(y))
#total_loss_of_the_batch = tf.reduce_sum(masked_losses)

# Calculate mean loss
# sum loss of one sequence divided by its sequence length (real points)
mean_loss_by_example = tf.reduce_sum(masked_losses, 1) / seq_length_of_the_batch 
mean_loss = tf.reduce_mean(mean_loss_by_example)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mean_loss)
init = tf.initialize_all_variables()

'''
Training our model
'''
with tf.Session() as sess:
    sess.run(init)  # initializer
    training_losses = []  # a list to store the loss of each epoch
    print("Start training process...")
    
    for i, epoch in enumerate(utils.gen_epochs(max_epoch, batch_size, train_data)):
        print("Epoch: %d/%d..." % (i + 1, max_epoch))
        training_loss = 0.0  # to add up the loss of each batch together 
        steps = 0  # the number of mini batch training in a epoch
        for batch_x, batch_y in epoch:  # randomly select mini batch from training data
            #print('batch_x: ',batch_x.shape)
            #print('batch_y: ',batch_y.shape)
            steps += 1
            mean_loss_, training_state, pred, _ = sess.run([mean_loss,
                                                     last_states,
                                                     prediction,
                                                     optimizer],
                                                     feed_dict = {x:batch_x, y:batch_y}
                                                     )
            print('predictions of the batch', pred)
            print("Average Loss for Batch: %d is %g" % (steps, mean_loss_)) # num_batch is calculated by data_length/batch_size
            training_loss += mean_loss_  # add up the mean loss of each batch
        mean_loss_epoch = training_loss / steps # mean loss of the epoch
        print("Average training loss for Epoch", i+1, ":", mean_loss_epoch)
        training_losses.append(mean_loss_epoch)  # store mean loss of each epoch in a list
    print("Finished training!")







