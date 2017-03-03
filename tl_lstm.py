#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time

"""Example of Synced sequence input and output.
This is a reimpmentation of the TensorFlow official PTB example in :
tensorflow/models/rnn/ptb
The batch_size can be seem as how many concurrent computations.\n
As the following example shows, the first batch learn the sequence information by using 0 to 9.\n
The second batch learn the sequence information by using 10 to 19.\n
So it ignores the information from 9 to 10 !\n
If only if we set the batch_size = 1, it will consider all information from 0 to 20.\n
The meaning of batch_size here is not the same with the MNIST example. In MNIST example,
batch_size reflects how many examples we consider in each iteration, while in
PTB example, batch_size is how many concurrent processes (segments)
for speed up computation.
Some Information will be ignored if batch_size > 1, however, if your dataset
is "long" enough (a text corpus usually has billions words), the ignored
information would not effect the final result.
In PTB tutorial, we setted batch_size = 20, so we cut the dataset into 20 segments.
At the begining of each epoch, we initialize (reset) the 20 RNN states for 20
segments, then go through 20 segments separately.
The training data will be generated as follow:\n
>>> train_data = [i for i in range(20)]
>>> for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
>>>     x, y = batch
>>>     print(x, '\n',y)
... [[ 0  1  2] <---x                       1st subset/ iteration
...  [10 11 12]]
... [[ 1  2  3] <---y
...  [11 12 13]]
...
... [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
...  [13 14 15]] <--- 2nd batch input
... [[ 4  5  6]  <--- 1st batch target
...  [14 15 16]] <--- 2nd batch target
...
... [[ 6  7  8]                             3rd subset/ iteration
...  [16 17 18]]
... [[ 7  8  9]
...  [17 18 19]]
Hao Dong: This example can also be considered as pre-training of the word
embedding matrix.
About RNN
----------
$ Karpathy Blog : http://karpathy.github.io/2015/05/21/rnn-effectiveness/
More TensorFlow official RNN examples can be found here
---------------------------------------------------------
$ RNN for PTB : https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks
$ Seq2seq : https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence-models
$ translation : tensorflow/models/rnn/translate
tensorflow (0.9.0)
"""

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
A) use the zero_state function on the cell object
B) for an rnn, all time steps share weights. We use one matrix to keep all
gate weights. Split by column into 4 parts to get the 4 gate weight matrices.
"""

flags = tf.flags
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
FLAGS = flags.FLAGS


def main(_):
    """
    The core of the model consists of an LSTM cell that processes one word at
    a time and computes probabilities of the possible continuations of the
    sentence. The memory state of the network is initialized with a vector
    of zeros and gets updated after reading each word. Also, for computational
    reasons, we will process data in mini-batches of size batch_size.
    """

    if FLAGS.model == "small":
        init_scale = 0.1
        learning_rate = 0.1
        max_grad_norm = 5
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 1
        print('hahaha')
    elif FLAGS.model == "medium":
        init_scale = 0.05
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 35
        hidden_size = 650
        max_epoch = 6
        max_max_epoch = 39
        keep_prob = 0.5
        lr_decay = 0.8
        batch_size = 20
        vocab_size = 1
    elif FLAGS.model == "large":
        init_scale = 0.04
        learning_rate = 1.0
        max_grad_norm = 10
        num_layers = 2
        num_steps = 35
        hidden_size = 1500
        max_epoch = 14
        max_max_epoch = 55
        keep_prob = 0.35
        lr_decay = 1 / 1.15
        batch_size = 20
        vocab_size = 1
    
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

    # Load PTB dataset
    train_data, valid_data, test_data, data_dict = load_phm_dataset()
    # train_data = train_data[0:int(100000/5)]    # for fast testing
    print('len(train_data) {}'.format(len(train_data))) # 929589 a list of int
    print('len(valid_data) {}'.format(len(valid_data))) # 73760  a list of int
    print('len(test_data)  {}'.format(len(test_data)))  # 82430  a list of int
    #print('vocab_size      {}'.format(vocab_size))      # 10000

    sess = tf.InteractiveSession()

    # One int represents one word, the meaning of batch_size here is not the
    # same with MNIST example, it is the number of concurrent processes for
    # computational reasons.

    # Training and Validing
    input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    # Testing (Evaluation)
    input_data_test = tf.placeholder(tf.int32, [1, 1])
    targets_test = tf.placeholder(tf.int32, [1, 1])

    def inference(x, is_training, num_steps, reuse=None, data_name = None):
        """If reuse is True, the inferences use the existing parameters,
        then different inferences share the same parameters.
        Note :
        - For DynamicRNNLayer, you can set dropout and the number of RNN layer internally.
        """
        print("\nnum_steps : %d, is_training : %s, reuse : %s" %
                                                (num_steps, is_training, reuse))
        initializer = tf.random_uniform_initializer(init_scale, init_scale)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.Index_2_Data_Inputlayer(
                        data_dict,
                        inputs = x,
                        name ='embedding',
                        data_name = data_name)
            network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='drop1')
            network = tl.layers.RNNLayer(network,
                        cell_fn=tf.contrib.rnn.BasicLSTMCell, #tf.nn.rnn_cell.BasicLSTMCell,
                        cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                        n_hidden=hidden_size,
                        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                        n_steps=num_steps,
                        return_last=False,
                        name='basic_lstm1')
            lstm1 = network
            network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='drop2')
            network = tl.layers.RNNLayer(network,
                        cell_fn=tf.contrib.rnn.BasicLSTMCell,#tf.nn.rnn_cell.BasicLSTMCell,
                        cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                        n_hidden=hidden_size,
                        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                        n_steps=num_steps,
                        return_last=False,
                        return_seq_2d=True,
                        name='basic_lstm2')
            lstm2 = network
            # Alternatively, if return_seq_2d=False, in the above RNN layer,
            # you can reshape the outputs as follow:
            # network = tl.layers.ReshapeLayer(network,
            #       shape=[-1, int(network.outputs._shape[-1])], name='reshape')
            network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='drop3')
            network = tl.layers.DenseLayer(network,
                        n_units=vocab_size,
                        W_init=tf.random_uniform_initializer(-init_scale, init_scale),
                        b_init=tf.random_uniform_initializer(-init_scale, init_scale),
                        act = tf.identity, name='output')
        return network, lstm1, lstm2

    # Inference for Training
    network, lstm1, lstm2 = inference(input_data,
                            is_training=True, num_steps=num_steps, reuse=None, data_name = "TrainInput")
    # Inference for Validating
    network_val, lstm1_val, lstm2_val = inference(input_data,
                            is_training=False, num_steps=num_steps, reuse=True, data_name = "ValidInput")
    # Inference for Testing (Evaluation)
    network_test, lstm1_test, lstm2_test = inference(input_data_test,
                            is_training=False, num_steps=1, reuse=True, data_name = "TestInput")

    # sess.run(tf.initialize_all_variables())
    tl.layers.initialize_global_variables(sess)

    def loss_fn(outputs, targets):#, batch_size, num_steps):
        # See tl.cost.cross_entropy_seq()
        # Returns the cost function of Cross-entropy of two sequences, implement
        # softmax internally.
        # outputs : 2D tensor [batch_size*num_steps, n_units of output layer]
        # targets : 2D tensor [batch_size, num_steps], need to be reshaped.
        # batch_size : RNN batch_size, number of concurrent processes.
        # n_examples = batch_size * num_steps
        # so
        # cost is the averaged cost of each mini-batch (concurrent process).
        #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(  # loss = tf.nn.seq2seq.sequence_loss_by_example( # TF0.12
        #    [outputs],
        #    [tf.reshape(targets, [-1])],
        #    [tf.ones_like(tf.reshape(targets, [-1]), dtype=tf.float32)])
            # [tf.ones([batch_size * num_steps])])
        print(type(outputs))
        print(type(targets))
        targets = tf.cast(targets, dtype = 'float32')
        targets = tf.reshape(targets, (-1,1))
        loss = tf.square(tf.subtract(targets, outputs))
        cost = tf.reduce_mean(loss)
        return cost
    print('network output:',network.outputs)
    print('targets: ',targets)
    # Cost for Training
    cost = loss_fn(network.outputs, targets)#, batch_size, num_steps)
    # Cost for Validating
    cost_val = loss_fn(network_val.outputs, targets)#, batch_size, num_steps)
    # Cost for Testing (Evaluation)
    cost_test = loss_fn(network_test.outputs, targets_test)#, 1, 1)

    # Truncated Backpropagation for training
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    # sess.run(tf.initialize_all_variables())
    tl.layers.initialize_global_variables(sess)

    network.print_params()
    network.print_layers()
    tl.layers.print_all_variables()

    print("\nStart learning a language model by using PTB dataset")
    for i in range(max_max_epoch):
        # decreases the initial learning rate after several
        # epoachs (defined by ``max_epoch``), by multipling a ``lr_decay``.
        new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        sess.run(tf.assign(lr, learning_rate * new_lr_decay))

        # Training
        print("Epoch: %d/%d Learning rate: %.3f" % (i + 1, max_max_epoch, sess.run(lr)))
        epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
        start_time = time.time()
        costs = 0.0; iters = 0
        # reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        for step, (x, y) in enumerate(phm_iterator(train_data, data_dict,
                                                    batch_size, num_steps, name = "TrainInput")):
            feed_dict = {input_data: x, targets: y,
                        lstm1.initial_state.c: state1[0],
                        lstm1.initial_state.h: state1[1],
                        lstm2.initial_state.c: state2[0],
                        lstm2.initial_state.h: state2[1],
                        }
            # For training, enable dropout
            feed_dict.update( network.all_drop )
            _cost, state1_c, state1_h, state2_c, state2_h, _ = \
                                    sess.run([cost,
                                            lstm1.final_state.c,
                                            lstm1.final_state.h,
                                            lstm2.final_state.c,
                                            lstm2.final_state.h,
                                            train_op],
                                            feed_dict=feed_dict
                                            )
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)

            costs += _cost; iters += num_steps

            if step % (epoch_size // 10) == 0:
                print("%.3f perplexity: %g speed: %.0f wps" %
                    (step * 1.0 / epoch_size, costs/iters,
                    iters * batch_size / (time.time() - start_time)))
        train_perplexity = costs / iters
        print("Epoch: %d/%d Train Perplexity: %g" % (i + 1, max_max_epoch,
                                                            train_perplexity))

        # Validing
        start_time = time.time()
        costs = 0.0; iters = 0
        # reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1_val.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2_val.initial_state)
        for step, (x, y) in enumerate(phm_iterator(valid_data, data_dict,
                                                    batch_size, num_steps, name = "ValidInput")):
            feed_dict = {input_data: x, targets: y,
                        lstm1_val.initial_state.c: state1[0],
                        lstm1_val.initial_state.h: state1[1],
                        lstm2_val.initial_state.c: state2[0],
                        lstm2_val.initial_state.h: state2[1],
                        }
            _cost, state1_c, state1_h, state2_c, state2_h, _ = \
                                    sess.run([cost_val,
                                            lstm1_val.final_state.c,
                                            lstm1_val.final_state.h,
                                            lstm2_val.final_state.c,
                                            lstm2_val.final_state.h,
                                            tf.no_op()],
                                            feed_dict=feed_dict
                                            )
            state1 = (state1_c, state1_h)
            state2 = (state2_c, state2_h)
            costs += _cost; iters += num_steps
        valid_perplexity = costs / iters
        print("Epoch: %d/%d Valid Perplexity: %g" % (i + 1, max_max_epoch,
                                                            valid_perplexity))

    print("Evaluation")
    # Testing
    # go through the test set step by step, it will take a while.
    start_time = time.time()
    costs = 0.0; iters = 0
    # reset all states at the begining
    state1 = tl.layers.initialize_rnn_state(lstm1_test.initial_state)
    state2 = tl.layers.initialize_rnn_state(lstm2_test.initial_state)
    for step, (x, y) in enumerate(phm_iterator(test_data,data_dict,
                                            batch_size=1, num_steps=1, name = "TestInput")):
        feed_dict = {input_data_test: x, targets_test: y,
                    lstm1_test.initial_state.c: state1[0],
                    lstm1_test.initial_state.h: state1[1],
                    lstm2_test.initial_state.c: state2[0],
                    lstm2_test.initial_state.h: state2[1],
                    }
        _cost, state1_c, state1_h, state2_c, state2_h = \
                                sess.run([cost_test,
                                        lstm1_test.final_state.c,
                                        lstm1_test.final_state.h,
                                        lstm2_test.final_state.c,
                                        lstm2_test.final_state.h,
                                        ],
                                        feed_dict=feed_dict
                                        )
        state1 = (state1_c, state1_h)
        state2 = (state2_c, state2_h)
        costs += _cost; iters += 1
    test_perplexity = costs / iters
    print("Test Perplexity: %g took %.2fs" % (test_perplexity, time.time() - start_time))

def phm_iterator(raw_data,data_dict, batch_size, num_steps, name):
    """
    Generate a generator that iterates on a list of words, see PTB tutorial. Yields (Returns) the source contexts and
    the target context by the given batch_size and num_steps (sequence_length).\n
    see ``PTB tutorial``.

    e.g. x = [0, 1, 2]  y = [1, 2, 3] , when batch_size = 1, num_steps = 3,
    raw_data = [i for i in range(100)]

    In TensorFlow's tutorial, this generates batch_size pointers into the raw
    PTB data, and allows minibatch iteration along these pointers.

    Parameters
    ----------
    raw_data : a list
            the context in list format; note that context usually be
            represented by splitting by space, and then convert to unique
            word IDs.
    batch_size : int
            the batch size.
    num_steps : int
            the number of unrolls. i.e. sequence_length

    Yields
    ------
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

    Raises
    ------
    ValueError : if batch_size or num_steps are too high.

    Examples
    --------
    >>> train_data = [i for i in range(20)]
    >>> for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
    >>>     x, y = batch
    >>>     print(x, y)
    ... [[ 0  1  2] <---x                       1st subset/ iteration
    ...  [10 11 12]]
    ... [[ 1  2  3] <---y
    ...  [11 12 13]]
    ...
    ... [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
    ...  [13 14 15]] <--- 2nd batch input
    ... [[ 4  5  6]  <--- 1st batch target
    ...  [14 15 16]] <--- 2nd batch target
    ...
    ... [[ 6  7  8]                             3rd subset/ iteration
    ...  [16 17 18]]
    ... [[ 7  8  9]
    ...  [17 18 19]]

    Code References
    ----------------
    - ``tensorflow/models/rnn/ptb/reader.py``
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps:(i+1)*num_steps]
        yield (x, data_dict[name][:, -1][y])

def load_phm_dataset():
    """Penn TreeBank (PTB) dataset is used in many LANGUAGE MODELING papers,
    including "Empirical Evaluation and Combination of Advanced Language
    Modeling Techniques", "Recurrent Neural Network Regularization".

    It consists of 929k training words, 73k validation words, and 82k test
    words. It has 10k words in its vocabulary.

    In "Recurrent Neural Network Regularization", they trained regularized LSTMs
    of two sizes; these are denoted the medium LSTM and large LSTM. Both LSTMs
    have two layers and are unrolled for 35 steps. They initialize the hidden
    states to zero. They then use the final hidden states of the current
    minibatch as the initial hidden state of the subsequent minibatch
    (successive minibatches sequentially traverse the training set).
    The size of each minibatch is 20.

    The medium LSTM has 650 units per layer and its parameters are initialized
    uniformly in [−0.05, 0.05]. They apply 50% dropout on the non-recurrent
    connections. They train the LSTM for 39 epochs with a learning rate of 1,
    and after 6 epochs they decrease it by a factor of 1.2 after each epoch.
    They clip the norm of the gradients (normalized by minibatch size) at 5.

    The large LSTM has 1500 units per layer and its parameters are initialized
    uniformly in [−0.04, 0.04]. We apply 65% dropout on the non-recurrent
    connections. They train the model for 55 epochs with a learning rate of 1;
    after 14 epochs they start to reduce the learning rate by a factor of 1.15
    after each epoch. They clip the norm of the gradients (normalized by
    minibatch size) at 10.

    Returns
    --------
    train_data, valid_data, test_data, vocabulary size

    Examples
    --------
    >>> train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()

    Code References
    ---------------
    - ``tensorflow.models.rnn.ptb import reader``

    Download Links
    ---------------
    - `Manual download <http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz>`_
    """
    '''
    input: data_path of the train/validation/test data
    ouput: a list with the same length as data.shape[0], each integer in the list
           represents a data point in the data (1, num_features)
    '''
    # read in train/validation/test data from numpy or csv file
    # shape (data_length, num_features)
    data_path = '/home/cyril/TurboEngine_Dataset_NASA/'
    train_data = np.load(data_path +"train_160359_25.npy")
    valid_data = np.load(data_path +"valid_45918_25.npy")
    test_data = np.load(data_path +"test_104897_25.npy")
    
    # the num of data points in each of the data
    train_length = train_data.shape[0]
    valid_length = valid_data.shape[0]
    test_length = test_data.shape[0]
    
    # use integers in the list to represent each data point
    train_index = [index for index in range(train_length)]
    valid_index = [index for index in range(valid_length)]
    test_index = [index for index in range(test_length)]
    
    # store all the data in a data dictionary
    data_dict = {"TrainInput" : train_data,
                 "ValidInput" : valid_data,
                 "TestInput"  : test_data
                }
    return train_index, valid_index, test_index, data_dict


        
if __name__ == "__main__":
    tf.app.run()