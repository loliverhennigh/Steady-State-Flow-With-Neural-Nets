#!/usr/bin/env python

import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.flow_net as flow_net
from utils.experiment_manager import make_checkpoint_path
from tqdm import *

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # global step counter
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # make inputs
    boundary, sflow = flow_net.inputs_flow(FLAGS.batch_size) 
    # create and unrap network
    sflow_p = flow_net.inference_flow(boundary, FLAGS.keep_prob) 
    # calc error
    error = flow_net.loss_flow(sflow_p, sflow) 
    # train hopefuly 
    train_op = flow_net.train(error, FLAGS.learning_rate, global_step)
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    sess.run(init)
 
    # init from checkpoint
    saver_restore = tf.train.Saver(variables)
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt is not None:
      print("init from " + TRAIN_DIR)
      try:
         saver_restore.restore(sess, ckpt.model_checkpoint_path)
      except:
         tf.gfile.DeleteRecursively(TRAIN_DIR)
         tf.gfile.MakeDirs(TRAIN_DIR)
         print("there was a problem using variables in checkpoint, random init will be used instead")

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

    # calc number of steps left to run
    run_steps = FLAGS.max_steps - int(sess.run(global_step))
    for step in xrange(run_steps):
      current_step = sess.run(global_step)
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if current_step%100 == 0:
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

      if current_step%2000 == 0:
        time.sleep(60)
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, current_step) 
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
