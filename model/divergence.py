
import tensorflow as tf
import numpy as np

def int_shape(x):
  return list(map(int, x.get_shape()))

def _simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y

def _simple_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='SAME')
  return y

def spatial_divergence_2d(field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation
  field_shape = int_shape(field)

  # make weight for x divergence
  weight_x_np = np.zeros([3,1,2,1])
  weight_x_np[0,0,0,0] = -1.0/2.0
  weight_x_np[1,0,0,0] = 0.0 
  weight_x_np[2,0,0,0] = 1.0/2.0

  weight_x = tf.constant(np.float32(weight_x_np))

  # make weight for y divergence
  weight_y_np = np.zeros([1,3,2,1])
  weight_y_np[0,0,1,0] = -1.0/2.0
  weight_y_np[0,1,1,0] = 0.0 
  weight_y_np[0,2,1,0] = 1.0/2.0

  weight_y = tf.constant(np.float32(weight_y_np))

  # calc gradientes
  field = tf.concat(axis=1, values=[field[:,-2:-1], field, field[:,0:1]]) 
  field = tf.concat(axis=2, values=[field[:,:,-2:-1], field, field[:,:,0:1]])
  field_dx = _simple_conv_2d(field, weight_x)
  field_dy = _simple_conv_2d(field, weight_y)

  # divergence of field
  field_div = field_dx + field_dy

  # kill boundarys (this is not correct! I should use boundarys but for right now I will not)
  field_div = tf.abs(field_div[:,1:-2,1:-2,:])

  return field_div

def spatial_divergence_3d(field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation
  field_shape = int_shape(field)
  field = tf.reshape(field, [field_shape[0]*field_shape[1], field_shape[2], field_shape[3], field_shape[4], field_shape[5]])

  # make weight for x divergence
  weight_x_np = np.zeros([3,1,1,4,1])
  weight_x_np[0,0,0,0,0] = -1.0/2.0
  weight_x_np[1,0,0,0,0] = 0.0 
  weight_x_np[2,0,0,0,0] = 1.0/2.0

  weight_x = tf.constant(np.float32(weight_x_np))

  # make weight for y divergence
  weight_y_np = np.zeros([1,3,1,4,1])
  weight_y_np[0,0,0,1,0] = -1.0/2.0
  weight_y_np[0,1,0,1,0] = 0.0 
  weight_y_np[0,2,0,1,0] = 1.0/2.0

  weight_y = tf.constant(np.float32(weight_y_np))

  # make weight for z divergence
  weight_z_np = np.zeros([1,1,3,4,1])
  weight_z_np[0,0,0,2,0] = -1.0/2.0
  weight_z_np[0,0,1,2,0] = 0.0 
  weight_z_np[0,0,2,2,0] = 1.0/2.0

  weight_z = tf.constant(np.float32(weight_z_np))

  # calc gradientes
  field_dx = _simple_conv_3d(field, weight_x)
  field_dy = _simple_conv_3d(field, weight_y)
  field_dz = _simple_conv_3d(field, weight_z)

  # divergence of field
  field_div = field_dx + field_dy + field_dz

  # kill boundarys (this is not correct! I should use boundarys but for right now I will not)
  field_div = tf.abs(field_div[:,1:-2,1:-2,:])

  return field_div

