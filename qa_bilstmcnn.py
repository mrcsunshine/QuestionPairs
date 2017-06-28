import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np


class BILSTMCNN(object):
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
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
		# Reshape to (n_steps*batch_size, n_input)
		# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        self.embedded_chars1=tf.transpose(self.embedded_chars1,[1,0,2])
        self.embedded_chars1=tf.reshape(self.embedded_chars1,[-1,n_input])
        self.embedded_chars1=tf.split(self.embedded_chars1,n_steps,0)
        self.embedded_chars2=tf.transpose(self.embedded_chars2,[1,0,2])
        self.embedded_chars2=tf.reshape(self.embedded_chars2,[-1,n_input])
        self.embedded_chars2=tf.split(self.embedded_chars2,n_steps,0)
        self.embedded_chars3=tf.transpose(self.embedded_chars3,[1,0,2])
        self.embedded_chars3=tf.reshape(self.embedded_chars3,[-1,n_input])
        self.embedded_chars3=tf.split(self.embedded_chars3,n_steps,0)
        with tf.name_scope("fw"),tf.variable_scope("fw"):
            #print(tf.get_variable_scope().name)
            fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_fw_cell_m=rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bw"),tf.variable_scope("bw"):
            #print(tf.get_variable_scope().name)
            bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell_m = rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
        # Get lstm cell output
        #try:
        with tf.name_scope("biw"),tf.variable_scope("biw"):
            lstm1, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.embedded_chars1, dtype=tf.float32)
            print(lstm1)
        with tf.name_scope("biw2"),tf.variable_scope("biw",reuse=True):
            lstm2, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.embedded_chars2, dtype=tf.float32)
            print(lstm2)
        with tf.name_scope("biw3"),tf.variable_scope("biw",reuse=True):
            lstm3, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.embedded_chars3, dtype=tf.float32)
            print(lstm3)

        cnn_input1 = tf.reshape(tf.transpose(lstm1,[1, 0, 2]), [batch_size,sequence_length, embedding_size,1])
        cnn_input2 = tf.reshape(tf.transpose(lstm2,[1, 0, 2]), [batch_size,sequence_length, embedding_size,1])
        cnn_input3 = tf.reshape(tf.transpose(lstm3,[1, 0, 2]), [batch_size,sequence_length, embedding_size,1])
        # cnn_input1=tf.expand_dims(lstm1, -1)	
        # cnn_input2=tf.expand_dims(lstm2, -1)
        # cnn_input3=tf.expand_dims(lstm3, -1)
        print(cnn_input1)
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


