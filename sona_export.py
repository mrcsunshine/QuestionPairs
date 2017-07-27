import tensorflow as tf
import datahelper
import codecs
import operator
import numpy as np
import datetime

import os
import sys
import itertools
import logging
from distutils.version import LooseVersion
import tensorflow as tf
assert LooseVersion(tf.__version__) < LooseVersion("1.0"), "This Example is for TF versions <= 1.0"
import os, sys
import sonoma as sn
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_length", 100, "sequence length (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#eval Parameters
tf.flags.DEFINE_string("vocab_filepath", "../../data/vocab_new.txt", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("feature_filepath", "../data/qid_question_feature_2.txt", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("qapairs_filepath", "../data/qid_qapairs_new.txt", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "./ten12/model-40000", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("checkpoint_dir", "./", "Checkpoint directory from training run")

export_version = 1

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
if FLAGS.qapairs_filepath==None or FLAGS.feature_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()
	
session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
sess=tf.Session(config=session_conf)
checkpoint_file = FLAGS.model
#vocab=datahelper.read_vocab(FLAGS.vocab_filepath)
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
graph = tf.get_default_graph()
input1=graph.get_operation_by_name('input_x_1').outputs[0]
pool1=graph.get_operation_by_name('pooled_reshape_1').outputs[0]

input_tensor_bindings={"input_query": (input1, "Input matrix")}
output_tensor_bindings={"output vector": (pool1, "output query vector")}
  
saver.restore(sess, checkpoint_file)
sn.tf_exporter(sess,
                 # Name of Model
                 "QueryEncoder",
                 # Version
                 export_version,
                 # Description of exported model
                 "Sonoma Export CNN model for Query Encoder",
                 # Export Path
                 FLAGS.checkpoint_dir+"export/",
                 # List of Interfaces to export
                 [ (input_tensor_bindings, output_tensor_bindings) ],
                 # Vocabularies: None for this example
                 {})
