import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np


class LSTMCNN(object):
    def __init__(self, sequence_length, vocab_size, 
    embedding_size, hidden_units, l2_reg_lambda, batch_size,filter_sizes, num_filters):

      # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x2")
        self.input_x3 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        print("input_x_1: ", self.input_x1)
      # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
      # Embedding layer
        with tf.device('/cpu:0'),tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(W, self.input_x1)
            #self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(W, self.input_x2)
            self.embedded_chars3 = tf.nn.embedding_lookup(W, self.input_x3)
            #self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=hidden_units
        n_layers=1
		# #if timemajor=ture
        # # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # # Permuting batch_size and n_steps
		# # Reshape to (n_steps*batch_size, n_input)
		# # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # self.embedded_chars1=tf.transpose(self.embedded_chars1,[1,0,2])
        # self.embedded_chars1=tf.reshape(self.embedded_chars1,[-1,n_input])
        # self.embedded_chars1=tf.split(0,n_steps,self.embedded_chars1)
        # self.embedded_chars2=tf.transpose(self.embedded_chars2,[1,0,2])
        # self.embedded_chars2=tf.reshape(self.embedded_chars2,[-1,n_input])
        # self.embedded_chars2=tf.split(0,n_steps,self.embedded_chars2)
        # self.embedded_chars3=tf.transpose(self.embedded_chars3,[1,0,2])
        # self.embedded_chars3=tf.reshape(self.embedded_chars3,[-1,n_input])
        # self.embedded_chars3=tf.split(0,n_steps,self.embedded_chars3)
		
        with tf.name_scope("lstm1"),tf.variable_scope("lstm"):
            lstm_cell=rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_dp_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_ml_cell=rnn.MultiRNNCell([lstm_dp_cell]*n_layers, state_is_tuple=True)
			#output size will be [batchsize,sequence_len,num_units] 因为lstm输出是cnn输入，所以num_units=embedding_size,或者把cnn的filtershape改成num_units
            print('output size:'+str(lstm_ml_cell.output_size))
            init_state = lstm_ml_cell.zero_state(batch_size, dtype=tf.float32)
            lstm1, lstm_whole1 = tf.nn.dynamic_rnn(lstm_ml_cell, inputs=self.embedded_chars1, initial_state=init_state, time_major=False)
            # lstm2, lstm_whole2 = tf.nn.dynamic_rnn(lstm_ml_cell, inputs=self.embedded_chars2, initial_state=init_state, time_major=False)			
            # lstm3, lstm_whole3 = tf.nn.dynamic_rnn(lstm_ml_cell, inputs=self.embedded_chars3, initial_state=init_state, time_major=False)						
            print('lstm1 output:',lstm1)
        with tf.name_scope("lstm2"),tf.variable_scope("lstm",reuse=True):
            # lstm_cell=rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            # lstm_dp_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.dropout_keep_prob)
            # lstm_ml_cell=rnn.MultiRNNCell([lstm_dp_cell]*n_layers, state_is_tuple=True)
			# #output size will be [batchsize,sequence_len,num_units] 因为lstm输出是cnn输入，所以num_units=embedding_size,或者把cnn的filtershape改成num_units
            # print('output size:'+str(lstm_ml_cell.output_size))
            init_state = lstm_ml_cell.zero_state(batch_size, dtype=tf.float32)
            lstm2, lstm_whole2 = tf.nn.dynamic_rnn(lstm_ml_cell, inputs=self.embedded_chars2, initial_state=init_state, time_major=False)
            print('lstm2 output:',lstm2)
        with tf.name_scope("lstm3"),tf.variable_scope("lstm",reuse=True):
            # lstm_cell=rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            # lstm_dp_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.dropout_keep_prob)
            # lstm_ml_cell=rnn.MultiRNNCell([lstm_dp_cell]*n_layers, state_is_tuple=True)
			# #output size will be [batchsize,sequence_len,num_units] 因为lstm输出是cnn输入，所以num_units=embedding_size,或者把cnn的filtershape改成num_units
            # print('output size:'+str(lstm_ml_cell.output_size))
            init_state = lstm_ml_cell.zero_state(batch_size, dtype=tf.float32)
            lstm3, lstm_whole3 = tf.nn.dynamic_rnn(lstm_ml_cell, inputs=self.embedded_chars3, initial_state=init_state, time_major=False)
            print('lstm3 output:',lstm3)
            print('lstm whole output:',lstm_whole3)
			
        cnn_input1=tf.expand_dims(lstm1, -1)	
        cnn_input2=tf.expand_dims(lstm2, -1)
        cnn_input3=tf.expand_dims(lstm3, -1)
        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
		
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    cnn_input1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
                )
                pooled_outputs_1.append(pooled)

                conv = tf.nn.conv2d(
                    cnn_input2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
                )
                pooled_outputs_2.append(pooled)

                conv = tf.nn.conv2d(
                    cnn_input3,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-3"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-3"
                )
                pooled_outputs_3.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        pooled_reshape_1 = tf.reshape(tf.concat(pooled_outputs_1,3), [-1, num_filters_total]) 
        pooled_reshape_2 = tf.reshape(tf.concat(pooled_outputs_2,3), [-1, num_filters_total]) 
        pooled_reshape_3 = tf.reshape(tf.concat(pooled_outputs_3,3), [-1, num_filters_total]) 
        #dropout
        pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        pooled_flat_3 = tf.nn.dropout(pooled_reshape_3, self.dropout_keep_prob)

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) #计算向量长度Batch模式
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        pooled_len_3 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_3, pooled_flat_3), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) #计算向量的点乘Batch模式
        pooled_mul_13 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_3), 1)

        with tf.name_scope("output"):
            self.cos_12 = tf.divide(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") #计算向量夹角Batch模式
            self.cos_13 = tf.divide(pooled_mul_13, tf.multiply(pooled_len_1, pooled_len_3))

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.05, shape=[batch_size], dtype=tf.float32)  #ori 0.05
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.cos_12, self.cos_13)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")


