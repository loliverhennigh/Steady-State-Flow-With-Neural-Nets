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

tf.app.flags.DEFINE_string('test_set', "car",
                            """ just car for now """)

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
    boundary_op = tf.placeholder(tf.float32, [1, shape[0], shape[1], 1])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    sflow_p = flow_net.inference_flow(boundary_op,1.0)
    #velocity_x, velocity_y = flow_net.velocity(sflow_p)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(TEST_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    for run in filenames:
      # read in boundary
      flow_name = run + '/fluid_flow_0002.h5'
      boundary_np = load_boundary(flow_name, shape).reshape([1, shape[0], shape[1], 1])
      sflow_true = load_state(flow_name, shape)
 
      # calc logits 
      sflow_generated = sess.run(sflow_p,feed_dict={boundary_op: boundary_np})[0]
      v_x, v_y, drag_x_g, drag_y_g, weird_bounds_g = sess.run([velocity_x, velocity_y, drag_x, drag_y, weird_bounds],feed_dict={boundary_op: boundary_np})
      print("drag x is " + str(drag_x_g))
      print("drag y is " + str(drag_y_g))

      if FLAGS.display_test: 
        # convert to display 
        sflow_plot = np.concatenate([sflow_true, sflow_generated, sflow_true - sflow_generated], axis=1) 
        boundary_concat = np.concatenate(3*[boundary_np], axis=2) 
        #sflow_plot = np.sqrt(np.square(sflow_plot[:,:,0]) + np.square(sflow_plot[:,:,1])+ np.square(sflow_plot[:,:,2])+ np.square(sflow_plot[:,:,3]) + np.square(sflow_plot[:,:,4]) + np.square(sflow_plot[:,:,5]) + np.square(sflow_plot[:,:,6]) + np.square(sflow_plot[:,:,7]) + np.square(sflow_plot[:,:,8])) - .05 *boundary_concat[0,:,:,0]
        #sflow_plot = sflow_plot[:,:,0]
        #print(weird_bounds_g.shape)
        #sflow_plot = weird_bounds_g[0,:,:]
        #sflow_plot = weird_bounds_g[0,:,:] - .05 *boundary_np[0,1:-2,1:-1,0]
        sflow_plot = v_y[0,:,:]


        # display it
        plt.imshow(sflow_plot)
        plt.colorbar()
        plt.show()

    print("the percent error on " + FLAGS.test_set + " is")
    print(p_error)

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
