
import tensorflow as tf
import numpy as np

# code written for 3d case too

def int_shape(x):
  return list(map(int, x.get_shape()))

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def simple_trans_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(k)[2]]) 
  y = tf.nn.conv2d_transpose(x, k, output_shape, [1, 1, 1, 1], padding='SAME')
  y = tf.reshape(y, [int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2]), int(k.get_shape()[2])])
  return y

def simple_trans_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(k)[3]]) 
  y = tf.nn.conv3d_transpose(x, k, output_shape, [1, 1, 1, 1, 1], padding='SAME')
  y = tf.reshape(y, [int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2]), int(x.get_shape()[3]), int(k.get_shape()[3])])
  return y

def make_params_op(shape, boundary_set):
  params_op_set = tf.placeholder(tf.float32, [1, shape[0], shape[1], len(boundary_set)])
  params_op = tf.Variable(np.zeros((1, shape[0], shape[1], len(boundary_set))).astype(dtype=np.float32), name="params")
  params_op_init = tf.group(params_op.assign(params_op_set))
  return params_op, params_op_init, params_op_set

def make_params_train_op(loss, params, noise_std=0.01):
  # get shape
  params_shape = int_shape(params)
 
  # get train variables
  train_variables = tf.all_variables()
  train_variables = [variable for i, variable in enumerate(train_variables) if "params" in variable.name[:variable.name.index(':')]]

  # compute gradients
  grads = tf.gradients(loss, train_variables)

  # add noise possibly
  grads = gaussian_noise_layer(grads, noise_std)
 
  # find gradients just outside of params
  params_out = get_params_out(params)
  #grad_out = (tf.reshape(grads[0], [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]]) * tf.reshape(params_out, [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]])) - tf.reshape(params_out, [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]])
  grad_out = (tf.reshape(grads[0], [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]])) - 10000.0 * tf.reshape(params_out, [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]])
 
  # find top gradients 
  _up, index_up   = tf.nn.top_k(-grad_out,1)

  # grad up
  grad_up   = tf.reshape(tf.reduce_sum(tf.one_hot(index_up[0], params_shape[0]*params_shape[1]*params_shape[2]*params_shape[3]), axis=0), [params_shape[0], params_shape[1], params_shape[2], params_shape[3]])

  # find where gradient came from
  params_in = get_params_out(grad_up) 
  params_in = params_in * params
  #grad_in = (tf.reshape(grads[0], [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]]) * tf.reshape(params_in, [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]])) + tf.reshape(params_in, [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]])
  grad_in = (tf.reshape(grads[0], [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]]) + 10000.0 * tf.reshape(params_in, [params_shape[0], params_shape[1]*params_shape[2]*params_shape[3]]))


  # find top gradients 
  _do, index_down   = tf.nn.top_k(grad_in,1)

  # grads to apply
  grad_down   = tf.reshape(tf.reduce_sum(tf.one_hot(index_down[0], params_shape[0]*params_shape[1]*params_shape[2]*params_shape[3]), axis=0), [params_shape[0], params_shape[1], params_shape[2], params_shape[3]])

  # make opperation
  #train_op = params.assign(params + grad_up - grad_down) 
  train_op = params.assign(params + grad_up) 

  return train_op, grads[0]

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
  sphere_sizes = [5]
  square_sizes = []
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
  
def get_params_out(params):
  params_shape = int_shape(params)
  params = tf.reshape(params, params_shape + [1])
  edge_kernel = np.zeros((3,3,3,1,1)) + 1.0
  edge_kernel[1,1,1,0,0] = 0.0
  edge_kernel = tf.constant(edge_kernel, dtype=1)
  edge = simple_trans_conv_3d(params, edge_kernel)
  edge = edge * (-params + 1.0)
  edge = tf.reduce_sum(edge, axis=4)
  edge = tf.minimum(edge, 1.0)
  edge = tf.maximum(edge, 0.0)
  return edge


