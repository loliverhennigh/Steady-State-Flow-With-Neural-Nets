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
from utils.flow_reader import load_flow, load_boundary, load_state
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
tf.app.flags.DEFINE_bool('display_test', True,
                            """ display the test images """)
tf.app.flags.DEFINE_string('test_set', "car",
                            """ either car or random """)

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
  # get a list of image filenames
  filenames = glb('../data/computed_car_flow/*')
  filenames.sort(key=alphanum_key)
  filename_len = len(filenames)
  shape = [128, 256]

  with tf.Graph().as_default():
    # Make image placeholder
    boundary_op_set = tf.placeholder(tf.float32, [1, shape[0], shape[1], 1])
    boundary_op = tf.Variable(np.random.randint(2, size=(1, shape[0], shape[1], 1)).astype(dtype=np.float32), name="boundary")
    boundary_op_init = tf.group(boundary_op.assign(boundary_op_set))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    sflow_p = flow_net.inference(boundary_op,1.0)
    drag_x, drag_y, weird_bounds = flow_net.drag(boundary_op, sflow_p)
    boundary_out, boundary_in = flow_net.boundary_edge(boundary_op)
    boundary_in = boundary_op * boundary_in
    
    # train_op
    train_variables = tf.all_variables()
    train_variables = [variable for i, variable in enumerate(train_variables) if "boundary" in variable.name[:variable.name.index(':')]]
    grads = tf.gradients(drag_y, train_variables)
    bound_grad_out = tf.reshape(-grads[0], [1, shape[0]*shape[1]]) + tf.reshape(boundary_out, [1, shape[0]*shape[1]])
    bound_grad_in = tf.reshape(-grads[0], [1, shape[0]*shape[1]]) - tf.reshape(boundary_in, [1, shape[0]*shape[1]])
    _, index_up   = tf.nn.top_k( bound_grad_out,2)
    _, index_down = tf.nn.top_k(-bound_grad_in,1)
    grad_up   = tf.reshape(tf.reduce_sum(tf.one_hot(  index_up[0], shape[0]*shape[1]), axis=0), [1, shape[0], shape[1], 1])
    grad_down = tf.reshape(tf.reduce_sum(tf.one_hot(index_down[0], shape[0]*shape[1]), axis=0), [1, shape[0], shape[1], 1])
    #train_step = tf.group(boundary_op.assign(tf.minimum(tf.maximum(boundary_op + grad_up - grad_down, 0.0), 1.0)))
    train_step = tf.group(boundary_op.assign(boundary_op + grad_up - grad_down))
    #train_step = tf.group(boundary_op.assign(boundary_op - grad_down))

    # init graph
    init = tf.global_variables_initializer()

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    variables_to_restore = [variable for i, variable in enumerate(variables_to_restore) if "boundary" not in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(TEST_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    for run in filenames:
      # read in boundary
      flow_name = run + '/fluid_flow_0002.h5'
      boundary_np = load_boundary(flow_name, shape).reshape([1, shape[0], shape[1], 1])
      sess.run(boundary_op_init, feed_dict={boundary_op_set: boundary_np})

      for i in xrange(1000):
        #index_up_g = sess.run([grad_up], feed_dict={})[0]
        d_y, _ = sess.run([drag_y, train_step], feed_dict={})
        print(d_y)

      # calc logits 
      sflow_generated, boundary_generated = sess.run([sflow_p, boundary_op],feed_dict={})
      boundary_out_g = sess.run(boundary_in,feed_dict={})

      # convert to display 
      sflow_plot = sflow_generated[0]
      sflow_plot_1 = np.sqrt(np.square(sflow_plot[:,:,0]) + np.square(sflow_plot[:,:,1])+ np.square(sflow_plot[:,:,2])+ np.square(sflow_plot[:,:,3]) + np.square(sflow_plot[:,:,4]) + np.square(sflow_plot[:,:,5]) + np.square(sflow_plot[:,:,6]) + np.square(sflow_plot[:,:,7]) + np.square(sflow_plot[:,:,8]))
      #sflow_plot = np.sqrt(np.square(sflow_plot[:,:,0]) + np.square(sflow_plot[:,:,1])+ np.square(sflow_plot[:,:,2])+ np.square(sflow_plot[:,:,3]) + np.square(sflow_plot[:,:,4]) + np.square(sflow_plot[:,:,5]) + np.square(sflow_plot[:,:,6]) + np.square(sflow_plot[:,:,7]) + np.square(sflow_plot[:,:,8])) - 1. *boundary_generated[0,:,:,0] + .5 * boundary_np[0,:,:,0]
      #sflow_plot = 1. *boundary_generated[0,:,:,0] + .5 * boundary_np[0,:,:,0]
      #sflow_plot = 1. *boundary_out_g[0,:,:,0]
      sflow_plot_2 = 1. *boundary_out_g[0,:,:,0] + .5 * boundary_np[0,:,:,0]
      

      # display it
      fig = plt.figure()
      a = fig.add_subplot(1,2,1)
      plt.imshow(sflow_plot_1)
      a = fig.add_subplot(1,2,2)
      plt.imshow(sflow_plot_2)
      plt.colorbar()
      plt.show()


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
