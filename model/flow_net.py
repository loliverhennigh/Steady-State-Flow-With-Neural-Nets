
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import network_architecture
import input.flow_input as flow_input
import boundary_utils as boundary_utils
import LatFlow.Domain as dom
import inflow

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process and model.

# Training params
tf.app.flags.DEFINE_string('base_dir_flow', '../checkpoints_flow',
                            """dir to store trained flow net """)
tf.app.flags.DEFINE_string('base_dir_boundary', '../checkpoints_boundary',
                            """dir to store trained net boundary """)
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps',  300000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.9522,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """ r dropout """)
tf.app.flags.DEFINE_string('shape', '128x512',
                            """ shape of flow """)

# model params flow
tf.app.flags.DEFINE_string('flow_model', 'residual_network',
                           """ model name to train """)
tf.app.flags.DEFINE_integer('filter_size', 16,
                           """ filter size of first res block (preceding layers have double the filter size) """)
tf.app.flags.DEFINE_integer('nr_downsamples', 3,
                           """ number of downsamples in u network """)
tf.app.flags.DEFINE_integer('nr_residual_blocks', 2,
                           """ number of res blocks after each downsample """)
tf.app.flags.DEFINE_bool('gated_res', True,
                           """ gated resnet or not """)
tf.app.flags.DEFINE_string('nonlinearity', 'concat_elu',
                           """ nonlinearity used such as concat_elu, elu, concat_relu, relu """)
tf.app.flags.DEFINE_float('div_constant', 1.0,
                            """ apply to the divergence constant """)
tf.app.flags.DEFINE_integer('lb_seq_length', 30,
                            """ number of steps taken by LB solver during training """)
tf.app.flags.DEFINE_float('nu', 0.05,
                            """ viscocity of fluid """)
tf.app.flags.DEFINE_float('density', 1.0,
                            """ density for fluid solver """)
tf.app.flags.DEFINE_string('inflow_type', "parabolic",
                            """ type of inflow in simulation """)
tf.app.flags.DEFINE_float('inflow_value', 0.1,
                            """ inflow velocity """)
tf.app.flags.DEFINE_string('lattice_type', "D2Q9",
                            """ lattice structure for simulation """)

# model params boundary
tf.app.flags.DEFINE_string('boundary_type', 'shapes',
                           """ type of boundarys to train on """)
tf.app.flags.DEFINE_string('boundary_model', 'fc_conv',
                           """ model name to train boundary network on """)
tf.app.flags.DEFINE_integer('nr_boundary_params', 39,
                            """ number of boundary paramiters """)

# params boundary learn
tf.app.flags.DEFINE_string('boundary_learn_loss', "drag_xy",
                            """ what to mimimize in the boundary learning stuff """)
tf.app.flags.DEFINE_float('boundary_learn_lr', 0.4,
                            """ learning rate when learning boundary """)
tf.app.flags.DEFINE_float('boundary_learn_steps', 500,
                            """ number of steps when learning boundary """)

# test params
tf.app.flags.DEFINE_string('test_set', "car",
                            """ either car or random """)

def inputs_flow(batch_size, shape):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  boundary = tf.placeholder(tf.float32, [batch_size] + shape + [5])
  tf.summary.image('boundarys', boundary[...,0:1])
  tf.summary.image('u', boundary[...,1:4])
  tf.summary.image('u_on', boundary[...,-1:])
  return boundary

def inputs_boundary(input_dims, batch_size, shape):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  length_input = tf.placeholder(tf.float32, [batch_size, input_dims])
  boundary = tf.placeholder(tf.float32, [batch_size] + shape + [1])
  tf.summary.image('boundarys', boundary)
  return length_input, boundary

def inputs_boundary_learn(batch_size=1):
  params_op_set = tf.placeholder(tf.float32, [batch_size, FLAGS.nr_boundary_params])
  params_op = tf.Variable(np.zeros((batch_size, FLAGS.nr_boundary_params)).astype(dtype=np.float32), name="params")
  params_op_init = tf.group(params_op.assign(params_op_set))
  return params_op, params_op_init, params_op_set

def feed_dict_flows(batch_size, shape):
  boundarys = []
  us = []
  u_ons = []
  boundary_creator = boundary_utils.get_boundary_creator(FLAGS.boundary_type)
  inflow_vector = inflow.get_inflow_vector(FLAGS.inflow_type)
  for i in xrange(batch_size):
    #boundarys.append(boundary_utils.make_rand_boundary(shape))
    boundarys.append(boundary_creator(shape))
    u, u_on = inflow_vector(shape, np.array([FLAGS.inflow_value, 0.0, 0.0]))
    us.append(u)
    u_ons.append(u_on)
  boundarys = np.concatenate(boundarys)
  us = np.concatenate(us)
  u_ons = np.concatenate(u_ons)
  boundarys = np.concatenate([boundarys,us,u_ons], axis=-1)
  return boundarys

def feed_dict_boundary(input_dims, batch_size, shape):
  length_input = np.random.rand(batch_size, input_dims)
  boundarys = []
  boundary_creator = boundary_utils.get_creator(FLAGS.boundary_type)
  for i in xrange(batch_size):
    boundarys.append(boundary_utils.make_boundary_circle(length_input[i], shape))
  boundarys = np.concatenate(boundarys)
  return length_input, boundarys

def inference_flow(boundary, keep_prob):
  """Builds network.
  Args:
    inputs: input to network 
    keep_prob: dropout layer
  """
  with tf.variable_scope("flow_network") as scope:
    if FLAGS.flow_model == "residual_u_network": 
      sflow_p = network_architecture.residual_u_network(boundary, density=FLAGS.density, start_filter_size=FLAGS.filter_size, nr_downsamples=FLAGS.nr_downsamples, nr_residual_per_downsample=FLAGS.nr_residual_blocks, nonlinearity=FLAGS.nonlinearity)
    elif FLAGS.flow_model == "residual_network": 
      sflow_p = network_architecture.conv_res(boundary, keep_prob=keep_prob)
  return sflow_p

def inference_boundary(length_input, shape):
  with tf.variable_scope("boundary_network") as scope:
      if FLAGS.boundary_model == "fc_conv":
        boundary = network_architecture.fc_conv(length_input, shape)
  tf.summary.image('boundarys_g', boundary)
  return boundary

def loss_flow(sflow_p, boundary, global_step):
  shape = boundary.get_shape()
  shape = [int(shape[1]), int(shape[2])]

  # initalize flow solver
  domain = dom.Domain(FLAGS.lattice_type, FLAGS.nu, shape, boundary[...,0:1])

  # unroll flow solver for a few steps
  #sflow_t_list = domain.Unroll(sflow_p, FLAGS.lb_seq_length, inflow.apply_flow, (boundary[...,1:4], boundary[...,-1:]))
  sflow_t_list = domain.Unroll(tf.stop_gradient(sflow_p), FLAGS.lb_seq_length, inflow.apply_flow, (boundary[...,1:4], boundary[...,-1:]))

  # mse between predicted flow and flow solver
  #loss_mse_predicted  = tf.nn.l2_loss(sflow_p - sflow_t_list[0])
  #loss_mse_predicted += tf.nn.l2_loss(sflow_p - sflow_t_list[1])
  
  # mse between every two states of flow solver
  #for i in xrange(FLAGS.lb_seq_length-2):
  #  loss_mse_predicted += tf.nn.l2_loss(sflow_t_list[i] - sflow_t_list[i+1])
  #  loss_mse_predicted += tf.nn.l2_loss(sflow_t_list[i] - sflow_t_list[i+2])
  #loss_mse_predicted += tf.nn.l2_loss(sflow_p - sflow_t_list[-1])
  loss_mse_predicted = tf.nn.l2_loss(sflow_p - sflow_t_list[-1])
  tf.summary.scalar('mse_predicted_loss', loss_mse_predicted)

  # normalize loss
  loss = (loss_mse_predicted)/FLAGS.batch_size
  tf.summary.scalar('total_loss', loss)

  # image summary
  tf.summary.image('sflow_p_x', domain.Vel[0][:,:,:,0:1])
  tf.summary.image('sflow_p_y', domain.Vel[0][:,:,:,1:2])
  return loss

def loss_boundary(true_boundary, generated_boundary):
  intersection = tf.reduce_sum(generated_boundary * true_boundary)
  loss = -(2. * intersection + 1.) / (tf.reduce_sum(true_boundary) + tf.reduce_sum(generated_boundary) + 1.)
  tf.summary.scalar('loss', loss)
  return loss

def train(total_loss, lr, train_type="flow_network", global_step=None, variables=None):
   if train_type == "flow_network" or train_type == "boundary_network":
     train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step)
   elif train_type == "boundary_params":
     train_op = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, var_list=variables)
   return train_op

