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
from tqdm import *

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.flow_net as flow_net 
import input.flow_input as flow_input
from model.lattice import *
from utils.flow_reader import load_flow, load_boundary, load_state
from utils.experiment_manager import make_checkpoint_path
from utils.plot_helper import grey_to_short_rainbow

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

# video init
shape = [128, 256]
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + '_2d_video_.mov', fourcc, 10, (2*shape[1], shape[0]), True)


FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS)
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary, FLAGS)
print(FLOW_DIR)
print(BOUNDARY_DIR)

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def make_params_op(batch_size=1):
  params_op_set = tf.placeholder(tf.float32, [batch_size, FLAGS.nr_boundary_params])
  params_op = tf.Variable(np.zeros((batch_size, FLAGS.nr_boundary_params)).astype(dtype=np.float32), name="params")
  params_op_init = tf.group(params_op.assign(params_op_set))
  return params_op, params_op_init, params_op_set

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
  batch_size=1

  with tf.Graph().as_default():
    # Make image placeholder
    params_op, params_op_init, params_op_set = make_params_op(batch_size)
    boundary = flow_net.inference_bounds(tf.sigmoid(params_op))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    sflow_p = flow_net.inference_flow(boundary,1.0)
    sflow_p = add_lattice(sflow_p)

    # quantities to optimize
    velocity = lattice_to_vel(sflow_p)
    velocity_norm = vel_to_norm(velocity)
    force_t = lattice_to_force(sflow_p, tf.round(boundary+.4))
    force = lattice_to_force(sflow_p, boundary)
    drag_x = tf.reduce_sum(force[:,:,:,0])
    drag_y = tf.reduce_sum(force[:,:,:,1])
    drag_y_t = tf.reduce_sum(force_t[:,:,:,1])
    boundary_area = tf.reduce_sum(boundary)

    # loss (change this however)
    loss_b = tf.nn.l2_loss(boundary_area-3000)/1000000.0
    #loss = drag_y + 1.0* drag_x
    #loss = drag_x
    #loss = -drag_y - drag_x + loss_b
    #loss = drag_y + loss_b + drag_x
    #loss = drag_y - drag_x
    loss = drag_x + drag_y
    #loss = drag_y + loss_b
    #loss = drag_x + drag_y
    #loss = drag_y

    # train_op
    variables_to_train = tf.all_variables()
    variables_to_train = [variable for i, variable in enumerate(variables_to_train) if "params" in variable.name[:variable.name.index(':')]]
    train_step = flow_net.train(loss, 0.5, variables=variables_to_train)
    #train_step = flow_net.train(loss, 0.0005, variables=variables_to_train)

    # init graph
    init = tf.global_variables_initializer()

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    variables_to_restore_boundary = [variable for i, variable in enumerate(variables_to_restore) if "boundary_network" in variable.name[:variable.name.index(':')]]
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver_boundary = tf.train.Saver(variables_to_restore_boundary)
    saver_flow = tf.train.Saver(variables_to_restore_flow)

    sess = tf.Session()
    sess.run(init)

    ckpt_boundary = tf.train.get_checkpoint_state(BOUNDARY_DIR)
    ckpt_flow = tf.train.get_checkpoint_state(FLOW_DIR)

    saver_boundary.restore(sess, ckpt_boundary.model_checkpoint_path)
    saver_flow.restore(sess, ckpt_flow.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    params_np = np.random.rand(batch_size,FLAGS.nr_boundary_params)
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})
    run_time = 1000
    plot_error = np.zeros((run_time))
    plot_drag_y = np.zeros((run_time))
    plot_drag_y_t = np.zeros((run_time))
    plot_drag_x = np.zeros((run_time))

    for i in tqdm(xrange(run_time)):
      l, _, d_y, d_x, d_y_t = sess.run([loss, train_step, drag_y, drag_x, drag_y_t], feed_dict={})
      if i > 0:
        plot_error[i] = l
        plot_drag_x[i] = d_x
        plot_drag_y[i] = d_y
        plot_drag_y_t[i] = d_y_t
      if i % 50 == 0:
        velocity_norm_g, boundary_g = sess.run([velocity_norm, boundary],feed_dict={})
        #velocity_norm_g, boundary_g = sess.run([velocity_norm, force],feed_dict={})
        sflow_plot = np.concatenate([ 5.0*velocity_norm_g[0], boundary_g[0]], axis=1)
        #sflow_plot = np.concatenate([10.0*velocity_norm_g[0], boundary_g[0,:,:,0]], axis=1)
        sflow_plot = np.uint8(grey_to_short_rainbow(sflow_plot))
        video.write(sflow_plot)

    video.release()
    cv2.destroyAllWindows()

    # calc logits 
    #velocity_norm_g, boundary_g = sess.run([velocity_norm, boundary],feed_dict={})

    # convert to display 
    velocity_norm_g = velocity_norm_g[0,:,:,0]
    boundary_g = boundary_g[0,:,:,0]
      
    # display it
    fig = plt.figure()
    a = fig.add_subplot(1,3,1)
    plt.imshow(velocity_norm_g)
    a = fig.add_subplot(1,3,2)
    plt.imshow(boundary_g)
    a = fig.add_subplot(1,3,3)
    plt.plot(plot_error, label="loss")
    plt.plot(plot_drag_x, label="drag_x")
    plt.plot(plot_drag_y, label="drag_y")
    plt.plot(plot_drag_y_t, label="drag_y_t")
    plt.legend()
    plt.colorbar()
    plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
