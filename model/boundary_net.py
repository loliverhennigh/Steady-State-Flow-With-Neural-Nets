
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import flow_architecture
import input.flow_input as flow_input

FLAGS = tf.app.flags.FLAGS

def inputs(input_dims, batch_size):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  length_input = tf.placeholder([batch_size, input_dims])
  return length_input

def inference(length_input):
  """Builds network.
  Args:
    inputs: input to network 
  """
  boundary = flow_architecture.fc_conv(length_input)
  return boundary

def loss_boundary(true_boundary, generated_boundary):
  # should change this loss to dice roll
  loss = tf.nn.l2_loss(true_boundary - generated_boundary)
  tf.summary.scalar('loss', loss)
  return loss

