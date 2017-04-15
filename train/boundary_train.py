
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.flow_net as flow_net
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints_boundary',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.7,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """ keep probability for dropout """)


TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)
print(TRAIN_DIR)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    input_dims = 9
    length_inputs, boundary_t = flow_net.inputs_bounds(input_dims, FLAGS.batch_size) 
    # create and unrap network
    boundary_g = flow_net.inference_bounds(length_inputs) 
    # calc error
    error = flow_net.loss_bounds(boundary_t, boundary_g) 
    # train hopefuly 
    train_op = flow_net.train(error, FLAGS.learning_rate)
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
    for i, variable in enumerate(variables):
      print '----------------------------------------------'
      print variable.name[:variable.name.index(':')]

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

    for step in xrange(FLAGS.max_steps):
      t = time.time()
      fd_length_inputs, fd_boundary = flow_net.feed_dict_bounds(input_dims, FLAGS.batch_size)
      _ , loss_value = sess.run([train_op, error],feed_dict={length_inputs: fd_length_inputs, boundary_t: fd_boundary})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        summary_str = sess.run(summary_op, feed_dict={length_inputs: fd_length_inputs, boundary_t: fd_boundary})
        summary_writer.add_summary(summary_str, step) 
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

      if step%1000 == 0:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
