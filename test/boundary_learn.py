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
from model.lattice import *
from utils.flow_reader import load_flow, load_boundary, load_state
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

# video init
shape = [128, 256]
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + '_2d_video_.mov', fourcc, 40, (2*shape[1], shape[0]), True)


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

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)
BOUNDARY_DIR = make_checkpoint_path('../checkpoints_boundary', FLAGS)

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def make_params_op():
  params_op_set = tf.placeholder(tf.float32, [1, 9])
  params_op = tf.Variable(np.zeros((1, 9)).astype(dtype=np.float32), name="params")
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

  with tf.Graph().as_default():
    # Make image placeholder
    params_op, params_op_init, params_op_set = make_params_op()
    boundary = flow_net.inference_bounds(params_op)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    sflow_p = flow_net.inference_flow(boundary,1.0)
    sflow_p = add_lattice(sflow_p)

    # quantities to optimize
    velocity = lattice_to_vel(sflow_p)
    velocity_norm = vel_to_norm(velocity)
    force = lattice_to_force(sflow_p, boundary)
    drag_x = tf.reduce_sum(force[:,:,:,0])
    drag_y = tf.reduce_sum(force[:,:,:,1])
    boundary_area = tf.reduce_sum(boundary)

    # loss (change this however)
    #loss = boundary_area
    #loss = drag_y + 2.* drag_x
    loss = drag_x
    #loss = drag_y

    # train_op
    variables_to_train = tf.all_variables()
    variables_to_train = [variable for i, variable in enumerate(variables_to_train) if "params" in variable.name[:variable.name.index(':')]]
    print(variables_to_train)
    train_step = flow_net.train(loss, 0.05, variables=variables_to_train)
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
    print(BOUNDARY_DIR)
    ckpt_flow = tf.train.get_checkpoint_state(FLOW_DIR)
    print(FLOW_DIR)

    saver_boundary.restore(sess, ckpt_boundary.model_checkpoint_path)
    saver_flow.restore(sess, ckpt_flow.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    params_np = np.random.rand(1,9)
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})

    for i in xrange(10):
      d_y, _, params = sess.run([loss, train_step, drag_y], feed_dict={})
      print(d_y)
      print(params)
      if i % 1 == 0:
        velocity_norm_g, boundary_g = sess.run([velocity_norm, boundary],feed_dict={})
        sflow_plot = np.concatenate([10.0*velocity_norm_g[0], boundary_g[0]], axis=1)
        sflow_plot = np.abs(np.concatenate(3*[sflow_plot], axis=2))
        sflow_plot = np.uint8(100.0*sflow_plot)
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
    a = fig.add_subplot(1,2,1)
    plt.imshow(velocity_norm_g)
    a = fig.add_subplot(1,2,2)
    plt.imshow(boundary_g)
    plt.colorbar()
    plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
