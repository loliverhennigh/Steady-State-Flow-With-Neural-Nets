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
import model.inflow as inflow
import LatFlow.Domain as dom
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

def make_car_boundary(shape, car_shape):
  img = cv2.imread("../cars/car_001.png", 0)
  img = cv2.flip(img, 1)
  resized_img = cv2.resize(img, car_shape)
  resized_img = -np.rint(resized_img/255.0).astype(int).astype(np.float32) + 1.0
  resized_img = resized_img.reshape([1, car_shape[1], car_shape[0], 1])
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, shape[0]-car_shape[1]:, 32:32+car_shape[0], :] = resized_img
  boundary[:,0,:,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  return boundary

def evaluate():
  """Run Eval once.
  """
  # get a list of image filenames
  filenames = glb('../data/computed_car_flow/*')
  filenames.sort(key=alphanum_key)
  filename_len = len(filenames)
  shape = [128, 512]
  #shape = [128, 256]
  #shape = [64, 128]
  #shape = [32, 64]

  with tf.Session() as sess:
    # Make image placeholder
    boundary_op = flow_net.inputs_flow(batch_size=1, shape=shape)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pyramid_boundary, pyramid_flow = flow_net.inference_flow(boundary_op,1.0)
    seq_length = 1000
    domain = dom.Domain(FLAGS.lattice_type, FLAGS.nu, shape, boundary_op[:,:,:,0:1])
    sflow_t = domain.Unroll(pyramid_flow[-1], 1, inflow.apply_flow, (boundary_op[...,1:4], boundary_op[...,-1:]))[-1]
    u_p = domain.Vel[0]
    sflow_t_list = domain.Unroll(sflow_t, seq_length-1, inflow.apply_flow, (boundary_op[...,1:4], boundary_op[...,-1:]))
    u_t = domain.Vel[0]
    #u_in = lb.make_u_input(shape)
    sflow_t = sflow_t_list[-1]

    # record diff
    diff = []
    diff.append(tf.nn.l2_loss((1.0-boundary_op[...,0:1])*(pyramid_flow[-1] - sflow_t_list[0])))
    for i in xrange(seq_length-2):
      diff.append(tf.nn.l2_loss((1.0-boundary_op[...,0:1])*(sflow_t_list[i+1] - sflow_t_list[i])))
    diff = tf.stack(diff)

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    variables_to_restore = tf.all_variables()
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore_flow)
    ckpt = tf.train.get_checkpoint_state(FLOW_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    #for run in filenames:
    for i in xrange(10):
      # read in boundary
      boundary_car = make_car_boundary(shape=shape, car_shape=(int(shape[1]/2.3), int(shape[0]/1.6)))
      boundary_np = flow_net.feed_dict_flows(1,shape)
      #boundary_np[:,:,:,0:1] = boundary_car
      #sflow_true = load_state(flow_name, [128,256])
      # calc logits 
      vel_p, vel_t = sess.run([u_p,u_t],feed_dict={boundary_op: boundary_np})
      diff_generated = sess.run(diff,feed_dict={boundary_op: boundary_np})
      plt.plot(diff_generated)
      plt.show()
      #vel_t = np.minimum(vel_t, 1.2)
      #vel_t = np.maximum(vel_t, -1.2)
      #sflow_plot = np.concatenate([vel_p, sflow_true, vel_p - sflow_true], axis=1)
      sflow_plot = np.concatenate([vel_p, vel_t, np.abs(vel_p - vel_t)], axis=1)
      #sflow_plot = vel_p
      sflow_plot = sflow_plot[0,:,:,0]

      # display it
      plt.imshow(sflow_plot)
      #plt.imshow(sflow_generated[0,:,:,0])
      plt.colorbar()
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
