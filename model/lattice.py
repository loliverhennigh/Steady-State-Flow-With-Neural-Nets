
import tensorflow as tf
import numpy as np

# code written for 3d case too

def int_shape(x):
  return list(map(int, x.get_shape()))

# easy to add more
VELOCITY_KERNEL_2D = np.zeros((3,3,2,1))
VELOCITY_KERNEL_2D[2,1,0,0] =  1.0
VELOCITY_KERNEL_2D[0,1,0,0] = -1.0
VELOCITY_KERNEL_2D[1,2,1,0] =  1.0
VELOCITY_KERNEL_2D[1,0,1,0] = -1.0
VELOCITY_KERNEL_3D = np.zeros((3,3,3,3,1))
VELOCITY_KERNEL_3D[2,1,1,2,0] =  1.0
VELOCITY_KERNEL_3D[0,1,1,2,0] = -1.0
VELOCITY_KERNEL_3D[1,2,1,1,0] =  1.0
VELOCITY_KERNEL_3D[1,0,1,1,0] = -1.0
VELOCITY_KERNEL_3D[1,1,2,0,0] =  1.0
VELOCITY_KERNEL_3D[1,1,0,0,0] = -1.0

BOUNDARY_EDGE_KERNEL_2D = np.zeros((3,3,9,1))
BOUNDARY_EDGE_KERNEL_2D[0,1,2,0] = 1.0 # up
BOUNDARY_EDGE_KERNEL_2D[1,0,1,0] = 1.0 # right
BOUNDARY_EDGE_KERNEL_2D[2,1,4,0] = 1.0 # down
BOUNDARY_EDGE_KERNEL_2D[1,2,3,0] = 1.0 # left
BOUNDARY_EDGE_KERNEL_2D[0,0,5,0] = 1.0 # up right
BOUNDARY_EDGE_KERNEL_2D[2,0,8,0] = 1.0 # down right
BOUNDARY_EDGE_KERNEL_2D[2,2,7,0] = 1.0 # down left
BOUNDARY_EDGE_KERNEL_2D[0,2,6,0] = 1.0 # up left
BOUNDARY_EDGE_KERNEL_3D = np.zeros((3,3,3,9,1))
BOUNDARY_EDGE_KERNEL_3D[0,1,1,1,0] = 1.0 # up
BOUNDARY_EDGE_KERNEL_3D[2,1,1,1,0] = 1.0 # down
BOUNDARY_EDGE_KERNEL_3D[1,0,1,1,0] = 1.0 # right
BOUNDARY_EDGE_KERNEL_3D[1,2,1,1,0] = 1.0 # left
BOUNDARY_EDGE_KERNEL_3D[1,1,0,1,0] = 1.0 # in
BOUNDARY_EDGE_KERNEL_3D[1,1,2,1,0] = 1.0 # out
BOUNDARY_EDGE_KERNEL_3D[0,0,0,1,0] = 1.0 # up right in
BOUNDARY_EDGE_KERNEL_3D[2,2,2,1,0] = 1.0 # down left out
BOUNDARY_EDGE_KERNEL_3D[0,0,2,1,0] = 1.0 # up right out
BOUNDARY_EDGE_KERNEL_3D[2,2,0,1,0] = 1.0 # down left in 
BOUNDARY_EDGE_KERNEL_3D[0,2,0,1,0] = 1.0 # up left in 
BOUNDARY_EDGE_KERNEL_3D[2,0,2,1,0] = 1.0 # down right out
BOUNDARY_EDGE_KERNEL_3D[0,2,2,1,0] = 1.0 # up left out
BOUNDARY_EDGE_KERNEL_3D[2,0,0,1,0] = 1.0 # down right in 

def simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def simple_trans_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(k)[2]]) 
  y = tf.nn.conv2d_transpose(x, k, output_shape, [1, 1, 1, 1], padding='SAME')
  y = tf.reshape(y, [int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2]), int(k.get_shape()[2])])
  return y

def simple_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='VALID')
  return y

def simple_trans_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(k)[2]]) 
  y = tf.nn.conv3d_transpose(x, k, [1, 1, 1, 1, 1], padding='VALID')
  y = tf.reshape(y, [x.get_shape()[0], x.get_shape()[1], x.get_shape()[2], k.get_shape()[2]])
  return y

def get_weights(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(np.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.]), dtype=1)
  elif lattice_size == 15:
    return tf.constant(np.array([2./9., 1./9., 1./9., 1./9., 1./9.,  1./9.,  1./9., 1./72., 1./72. , 1./72., 1./72., 1./72., 1./72., 1./72., 1./72.]), dtype=1)

def get_lveloc(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    #return tf.constant(np.array([ [0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1] ]), dtype=1)
    return tf.constant(np.array([ [0,0], [0,1], [1,0], [0,-1], [-1,0], [1,1], [1,-1], [-1,-1], [-1,1] ]), dtype=1)
  elif lattice_size == 15:
    return tf.constant(np.array([ [ 0, 0, 0], [ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1], [ 0, 0,-1], [ 1, 1, 1], [-1,-1,-1], [ 1, 1,-1], [-1,-1, 1], [ 1,-1, 1], [-1, 1,-1], [ 1,-1,-1], [-1, 1, 1] ]), dtype=1)

def get_opposite(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]), dtype=1)
  elif lattice_size == 15:
    return tf.constant(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]), dtype=1)

def get_velocity_kernel(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(VELOCITY_KERNEL_2D, dtype=1)
  elif lattice_size == 15:
    return tf.constant(VELOCITY_KERNEL_3D, dtype=1)

def get_edge_kernel(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(BOUNDARY_EDGE_KERNEL_2D, dtype=1)
  elif lattice_size == 15:
    return tf.constant(BOUNDARY_EDGE_KERNEL_3D, dtype=1)

def subtract_lattice(lattice):
  Weights = get_weights(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Weights = tf.reshape(Weights, dims*[1] + [int(Weights.get_shape()[0])])
  lattice = lattice - Weights
  return lattice

def add_lattice(lattice):
  Weights = get_weights(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Weights = tf.reshape(Weights, dims*[1] + [int(Weights.get_shape()[0])])
  lattice = lattice + Weights
  return lattice

def lattice_to_vel(lattice):
  # get velocity vector field from lattice
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Lveloc_shape = list(map(int, Lveloc.get_shape()))
  Lveloc = tf.reshape(Lveloc, dims*[1] + Lveloc_shape)
  lattice_shape = list(map(int, lattice.get_shape()))
  lattice = tf.reshape(lattice, lattice_shape + [1])
  velocity = tf.reduce_sum(Lveloc * lattice, axis=dims)
  return velocity

def vel_to_norm(velocity):
  if len(velocity.get_shape()) == 4:
    velocity_norm = tf.sqrt(tf.square(velocity[:,:,:,0:1]) + tf.square(velocity[:,:,:,1:2]))
  else:
    velocity_norm = tf.sqrt(tf.square(velocity[:,:,:,:,0:1]) + tf.square(velocity[:,:,:,:,1:2]) + tf.square(velocity[:,:,:,:,2:3]))
  return velocity_norm

def lattice_to_rho(lattice):
  dims = len(lattice.get_shape())-1
  rho = tf.reduce_sum(lattice, axis=dims)
  rho = tf.expand_dims(rho, axis=dims)
  return rho

def lattice_to_divergence(lattice):
  velocity = lattice_to_vel(lattice)
  velocity_shape = list(map(int, velocity.get_shape()))
  velocity_kernel = get_velocity_kernel(int(lattice.get_shape()[-1]))
  if len(velocity_shape) == 4:
    divergence = simple_conv_2d(velocity, velocity_kernel)
    divergence = divergence[:,1:-1,1:-1,:]
  else:
    divergence = simple_conv_3d(velocity, velocity_kernel)
    divergence = divergence[:,1:-1,1:-1,1:-1,:]
  return divergence

def lattice_to_flux(lattice, boundary):
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  rho = lattice_to_rho(lattice)
  velocity = lattice_to_vel(lattice)
  flux = velocity * rho * (-boundary + 1.0)
  return flux

def lattice_to_force(lattice, boundary):
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Lveloc_shape = list(map(int, Lveloc.get_shape()))
  Lveloc = tf.reshape(Lveloc, dims*[1] + Lveloc_shape)
  boundary_shape = list(map(int, boundary.get_shape()))
  boundary_edge_kernel = get_edge_kernel(int(lattice.get_shape()[-1]))
  if len(boundary.get_shape()) == 4:
    edge = simple_trans_conv_2d(boundary,boundary_edge_kernel) 
    edge = edge[:,1:-1,1:-1,:]
    boundary = boundary[:,1:-1,1:-1,:]
    lattice = lattice[:,1:-1,1:-1,:]
  else: 
    edge = simple_trans_conv_3d(boundary, boundary_edge_kernel)
    edge = edge[:,1:-1,1:-1,1:-1,:]
    boundary = boundary[:,1:-1,1:-1,1:-1,:]
  edge = edge * (-boundary + 1.0)
  edge = edge * lattice
  edge_shape = list(map(int, edge.get_shape()))
  edge = tf.reshape(edge, edge_shape + [1])
  force = tf.reduce_sum(edge * Lveloc, axis=dims)
  return force

def boundary_out(boundary):
  boundary_shape = list(map(int, boundary.get_shape()))
  boundary_edge_kernel = get_edge_kernel(9)
  if len(boundary.get_shape()) == 4:
    edge = simple_trans_conv_2d(boundary,boundary_edge_kernel)
  else:
    edge = simple_trans_conv_3d(boundary,boundary_edge_kernel)
  edge = edge * (-boundary + 1.0)
  edge = tf.reduce_sum(edge, axis=3)
  edge = edge - 2.0
  edge = tf.minimum(edge, 1.0)
  edge = tf.maximum(edge, 0.0)
  return edge

def boundary_in(boundary):
  boundary_shape = list(map(int, boundary.get_shape()))
  boundary_edge_kernel = get_edge_kernel(9)
  if len(boundary.get_shape()) == 4:
    edge = simple_trans_conv_2d((-boundary+1),boundary_edge_kernel)
  else:
    edge = simple_trans_conv_3d((-boundary+1),boundary_edge_kernel)
  edge = edge * boundary
  edge = tf.reduce_sum(edge, axis=3)
  edge = edge - 2.0
  #edge = edge 
  edge = tf.minimum(edge, 1.0)
  edge = tf.maximum(edge, 0.0)
  return edge



