#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from qa_lstmcnn import LSTMCNN
import operator
import codecs
import sys
import traceback

#print tf.__version__

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_length", 100, "sequence length (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("num_units", 200, "Dimensionality of hidden units (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")

# Training parameters

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every",500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")

vocab = data_helpers.read_vocab()
#vector= data_helpers.load_vectors()
alist = data_helpers.read_alist()
raw = data_helpers.read_raw()
#x_train_1, x_train_2, x_train_3 = data_helpers.load_data_6(vector, alist, raw, FLAGS.batch_size)
testList = data_helpers.load_test_and_vectors()
#embedding = data_helpers.read_array()
#vectors = ''
#print('x_train_1', np.shape(x_train_1))
print("Load done...")

#val_file = './query_question_test_random.txt'
#val_file = './partial_16500_test.txt'
#val_file = './query_question_test100_partial.txt'
val_file= '../seqlen100/rand/qq_test_head1500.txt'
#val_file = './test_sample.txt'
precision = './lstm.acc'
#precision = './sample_test.acc'
#x_val, y_val = data_deepqa.load_data_val()

# Training
# ==================================================

with tf.Graph().as_default():
  #with tf.device("/gpu:1"):
  with tf.device("/cpu:0"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = LSTMCNN(
            sequence_length=FLAGS.sequence_length,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            hidden_units=FLAGS.num_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        #optimizer = tf.train.GradientDescentOptimizer(1e-2)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        #sess.run(cnn.embedding_init,feed_dict={cnn.embedding_placeholder:embedding})
        
        def train_step(x_batch_1, x_batch_2, x_batch_3):
            """
            A single training step
            """
            # print(x_batch_1.shape)
            # print(x_batch_2.shape)
            # print(x_batch_3.shape)
            feed_dict = {
              model.input_x1: x_batch_1,
              model.input_x2: x_batch_2,
              model.input_x3: x_batch_3,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob
           }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step():
          scoreList = []
          i = int(0)         
          while True:
              x_test_1, x_test_2, x_test_3 = data_helpers.load_data_val_6(testList, vocab, i, FLAGS.batch_size)
              print("x_test_1 length:{}".format(len(x_test_1)))
              print("x_test_2 length: {}".format(len(x_test_2)))
              print("x_test_3 length: {}".format(len(x_test_3)))
              feed_dict = {
                model.input_x1: x_test_1,
                model.input_x2: x_test_2,
                model.input_x3: x_test_3,
                model.dropout_keep_prob: 1.0
              }
              batch_scores = sess.run([model.cos_12], feed_dict)
              for score in batch_scores[0]:
                  scoreList.append(score)
              i += FLAGS.batch_size
              print('index begin:{}\n'.format(i))
              if i >= len(testList):
                  print('index out if testlist')
                  break
          sessdict = {}
          index = int(0)
          file=codecs.open(val_file,encoding='utf-8')
          line=file.readline() 
          while line!='':
              items = line.strip().split(' ')
              if len(items)==4:
                  if (len(items[2].strip().split('_'))==101) and (len(items[3].strip().split('_'))==101):
                      qid = items[1].split(':')[1]
                      qid=int(qid)
                  if not qid in sessdict:
                      sessdict[qid] = []
                  sessdict[qid].append((scoreList[index], items[0]))
                  index += 1
              if index >= len(testList):
                  print('index out of testlist: {}\n'.format(index))
                  break
              if index >=len(scoreList):
                  print('index out of scorelist: {}\n'.format(index))
                  break
              line=file.readline()
          top1_lev1 = float(0)
          top1_lev0 = float(0)
          top2_lev1 = float(0)
          top2_lev0 = float(0)	
          top3_lev1 = float(0)
          top3_lev0 = float(0)		  
          of = open(precision, 'a')
          for k, v in sessdict.items():
              v.sort(key=operator.itemgetter(0), reverse=True)
              print('socre and flag list:{}'.format(v))
              score1, flag1 = v[0]
              if flag1 == '1':
                  top1_lev1 += 1
              if flag1 == '0':
                  top1_lev0 += 1
              score2, flag2 = v[1]
              if flag1 == '1' or flag2=='1':
                  top2_lev1 += 1
              if flag1 == '0' and flag2=='0':
                  top2_lev0 += 1
              score3, flag3 = v[1]
              if flag1 == '1' or flag2=='1'or flag3=='1':
                  top3_lev1 += 1
              if flag1 == '0' and flag2=='0'and flag3=='0':
                  top3_lev0 += 1
          of.write('to1lev1:' + str(top1_lev1) + '\n')
          of.write('top1lev0:' + str(top1_lev0) + '\n')
          print('top1_lev1 ' + str(top1_lev1))
          print('top1_lev0 ' + str(top1_lev0))
          of.write('to2lev1:' + str(top2_lev1) + '\n')
          of.write('top2lev0:' + str(top2_lev0) + '\n')
          print('top2_lev1 ' + str(top2_lev1))
          print('top2_lev0 ' + str(top2_lev0))
          of.write('to3lev1:' + str(top3_lev1) + '\n')
          of.write('top3lev0:' + str(top3_lev0) + '\n')
          print('top3_lev1 ' + str(top3_lev1))
          print('top3_lev0 ' + str(top3_lev0))
          of.close()

        # Generate batches
        # Training loop. For each batch...
        for i in range(FLAGS.num_epochs):
            try:
                x_batch_1, x_batch_2, x_batch_3 = data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
                train_step(x_batch_1, x_batch_2, x_batch_3)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step()
                    print("")
            except Exception as e:
                print(e)
                traceback.print_exc()
