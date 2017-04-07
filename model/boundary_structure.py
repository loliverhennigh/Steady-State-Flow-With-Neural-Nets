
import tensorflow as tf
import numpy as np

# code written for 3d case too

def int_shape(x):
  return list(map(int, x.get_shape()))

def simple_trans_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(k)[2]]) 
  y = tf.nn.conv2d_transpose(x, k, output_shape, [1, 1, 1, 1], padding='SAME')
  y = tf.reshape(y, [int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2]), int(k.get_shape()[2])])
  return y

def get_sphere_boundary(size_of_spheres):
  spheres = []
  for size in size_of_spheres:
    sphere = np.zeros((size, size, 1, 1))
    for i in xrange(size):
      for j in xrange(size):
        if np.sqrt(np.square(i-size/2.0) + np.square(j-size/2.0)) < size/2.0:
          sphere[i,j,0,0] = 1.0
    spheres.append(tf.constant(sphere, dtype=1))
  return spheres
 
def get_square_boundary(size_of_square):
  squares = []
  for size in size_of_square:
    square = np.zeros((size, size, 1, 1)) + 1.0
    squares.append(tf.constant(square, dtype=1))
  return squares

def make_boundary_set():
  # hard set for now
  sphere_sizes = [3,5,7,15,30]
  square_sizes = [3,5,7,15,30]
  spheres = get_sphere_boundary(sphere_sizes) 
  squares = get_square_boundary(square_sizes)
  shapes = spheres + squares
  return shapes

def params_to_boundary(params, boundary_set):
  for i in xrange(len(boundary_set)):
    if i == 0:
      boundary_add = simple_trans_conv_2d(params[:,:,:,i:i+1], boundary_set[i])
    else:
      boundary_add += simple_trans_conv_2d(params[:,:,:,i:i+1], boundary_set[i])
  boundary_add = tf.minimum(boundary_add, 1.0)
  return boundary_add
  

