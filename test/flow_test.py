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
import model.lattice as lat
import input.flow_input as flow_input
import model.lb_solver as lb
import model.divergence as divergence
from utils.flow_reader import load_flow, load_boundary, load_state
from utils.experiment_manager import make_checkpoint_path
import utils.boundary_utils as boundary_utils

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def evaluate():
  """Run Eval once.
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
    #sflow_p = lb.zeros_f(shape, density=1.0, solver_type="D2Q9")
    seq_length = 100
    u_in = lb.make_u_input(shape)
    sflow_t_list = lb.lbm_seq(sflow_p, boundary_op, u_in, seq_length, init_density=1.0, tau=1.0)
    sflow_t = sflow_t_list[-1]
    u_p = lb.f_to_u_full(sflow_p) 
    norm_u_p = tf.sqrt(tf.square(u_p[:,:,:,0:1]) + tf.square(u_p[:,:,:,1:2]))
    div_p = divergence.spatial_divergence_2d(u_p)
    u_t = lb.f_to_u_full(sflow_t)
    norm_u_t = tf.sqrt(tf.square(u_t[:,:,:,0:1]) + tf.square(u_t[:,:,:,1:2]))
    div_t = divergence.spatial_divergence_2d(u_t)

    # record diff
    diff = []
    for i in xrange(seq_length):
      diff.append(tf.nn.l2_loss(sflow_t_list[i] - sflow_p))
    diff = tf.stack(diff)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(FLOW_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    for run in filenames:
      # read in boundary
      #flow_name = run + '/fluid_flow_0002.h5'
      #boundary_np = load_boundary(flow_name, shape).reshape([1, shape[0], shape[1], 1])
      #sflow_true = load_state(flow_name, shape)
      boundary_np = boundary_utils.make_rand_boundary(shape)
      boundary_np = np.expand_dims(boundary_np, axis=0)
      boundary_np = np.expand_dims(boundary_np, axis=3)
 
      # calc logits 
      sflow_generated = sess.run(sflow_p,feed_dict={boundary_op: boundary_np})[0]
      vel_p, vel_t = sess.run([norm_u_p, norm_u_t],feed_dict={boundary_op: boundary_np})
      diff_generated = sess.run(diff,feed_dict={boundary_op: boundary_np})
      plt.plot(diff_generated)
      plt.show()
      print(np.sum(np.abs(vel_p - vel_t)))
      #vel_t = np.minimum(vel_t, 1.2)
      #vel_t = np.maximum(vel_t, -1.2)
      sflow_plot = np.concatenate([vel_p, vel_t, vel_p - vel_t], axis=1) 
      sflow_plot = sflow_plot[0,:,:,0]

      # display it
      plt.imshow(sflow_plot)
      plt.colorbar()
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
