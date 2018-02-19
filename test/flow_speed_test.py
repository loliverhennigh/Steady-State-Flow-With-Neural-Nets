from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import csv
import re
from glob import glob as glb

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.flow_net as flow_net 
import input.flow_input as flow_input
from utils.flow_reader import load_flow, load_boundary 
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.7,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """ keep probability for dropout """)

TEST_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  shape = [128, 256]

  with tf.Graph().as_default():
    # Make image placeholder
    boundary_op = tf.placeholder(tf.float32, [8, shape[0], shape[1], 1])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    sflow_p = flow_net.inference(boundary_op,1.0)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(TEST_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    num_runs = 1000
    boundary_np = np.zeros([8, shape[0], shape[1], 1])
    _ = sess.run(sflow_p,feed_dict={boundary_op: boundary_np})[0]
    t = time.time()
    for i in xrange(num_runs):
      # make boundary
      # calc logits 
      _ = sess.run(sflow_p,feed_dict={boundary_op: boundary_np})[0]

    elapsed = time.time() - t
    print("time per input is ")
    print(elapsed / (num_runs*8.))

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
