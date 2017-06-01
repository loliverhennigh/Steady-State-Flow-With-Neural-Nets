
import numpy as np
import tensorflow as tf
import divergence as divergence

def _make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=tf.float32)

def _simple_conv(x, k):
  """A simplified 2D convolution operation"""
  #if k.get_shape()[0] == 3:
  #  x = tf.pad(
  #      x, [[0, 0], [1, 1], [1, 1],
  #      [0, 0]], "REFLECT")
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def _transfer_kernel_D2Q9():
  transfer_kernel = np.zeros((3,3,9,9))
  # how to transfer states  
  transfer_kernel[1,0,0,0] = 1.0
  transfer_kernel[2,0,1,1] = 1.0
  transfer_kernel[2,1,2,2] = 1.0
  transfer_kernel[2,2,3,3] = 1.0
  transfer_kernel[1,2,4,4] = 1.0
  transfer_kernel[0,2,5,5] = 1.0
  transfer_kernel[0,1,6,6] = 1.0
  transfer_kernel[0,0,7,7] = 1.0
  # center
  transfer_kernel[1,1,8,8] = 1.0
  transfter_kernel = _make_kernel(transfer_kernel)
  return transfer_kernel

def _boundary_kernel_D2Q9():
  boundary_kernel = np.array([[ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
                           [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0]])
  boundary_kernel = np.expand_dims(boundary_kernel, axis=0)
  boundary_kernel = np.expand_dims(boundary_kernel, axis=0)
  boundary_kernel = tf.constant(boundary_kernel, dtype=tf.float32)
  return boundary_kernel

def _u_kernel_D2Q9():
  ux_kernel = np.array([ 1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0,  1.0,  0.0])
  ux_kernel = np.expand_dims(ux_kernel, axis=1)
  uy_kernel = np.array([ 0.0,  1.0,  1.0,  1.0,  0.0, -1.0, -1.0, -1.0,  0.0])
  uy_kernel = np.expand_dims(uy_kernel, axis=1)
  u_kernel = np.concatenate([ux_kernel, uy_kernel], axis=1)
  u_kernel = np.expand_dims(u_kernel, axis=0)
  u_kernel = np.expand_dims(u_kernel, axis=0)
  u_kernel = tf.constant(u_kernel, dtype=tf.float32)
  return u_kernel

def _weights_D2Q9():
  weights = tf.constant(np.array([1./9., 1./36., 1./9., 1./36., 1./9., 1./36., 1./9., 1./36., 4./9.]), dtype=1)
  weights = tf.reshape(weights, [1,1,1,9])
  return weights

def _create_boundary_cutter(boundary, solver_type="D2Q9"):
  if solver_type == "D2Q9":
    boundary_cutter = tf.concat( 8 * [boundary] + [tf.zeros_like(boundary)], axis=3)
    boundary_cutter_inv = (1.0 - boundary_cutter)
  return boundary_cutter, boundary_cutter_inv

def _set_velocity_boundary(f, u, density, pos="y_lower"):
  if pos == "y_lower":
    f_out = f[:,:,1:]
    f_edge = tf.split(f[:,:,0:1], 9, axis=3)
    f_edge[0] = f_edge[4] + (2.0/3.0)*density*u
    f_edge[1] = f_edge[5] + (1.0/6.0)*density*u - 0.5*(f_edge[2]-f_edge[6])
    f_edge[7] = f_edge[3] + (1.0/6.0)*density*u + 0.5*(f_edge[2]-f_edge[6])
    f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
    f_out = tf.concat([f_edge,f_out],axis=2)
  return f_out

def _pad_f(f):
  f_mobius = tf.concat(axis=1, values=[f[:,-2:-1], f, f[:,0:1]]) 
  f_mobius = tf.concat(axis=2, values=[f_mobius[:,:,-2:-1], f_mobius, f_mobius[:,:,0:1]])
  return f_mobius
  
def _propagate(f, propagate_kernel):
  f_mobius = _pad_f(f)
  f_mobius = _simple_conv(f_mobius, propagate_kernel)
  return f_mobius

def _bounce_back(f, boundary_cutter, boundary_kernel):
  f_boundary = tf.multiply(f, boundary_cutter)
  f_boundary = _simple_conv(f_boundary, boundary_kernel)
  return f_boundary 

def _u_to_feq(u, density):
  t1 = 4.0/9 
  t2 = 1.0/9 
  t3 = 1.0/36
  c_squ = 1.0/3
  ux = u[:,:,:,0:1]
  uy = u[:,:,:,1:2]
  u_squ = tf.square(ux) + tf.square(uy)
  u_c2 = ux + uy
  u_c4 = -ux + uy
  u_c6 = -u_c2
  u_c8 = -u_c4
  # now FEQ 
  # this could probably be heavelily optimized with conv layers but for now I will go with this
  FEQ_0 = t2*density*(1.0 + tf.div(ux, c_squ) + tf.multiply(tf.square(tf.div(ux, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_2 = t2*density*(1.0 + tf.div(uy, c_squ) + tf.multiply(tf.square(tf.div(uy, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_4 = t2*density*(1.0 - tf.div(ux, c_squ) + tf.multiply(tf.square(tf.div(ux, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_6 = t2*density*(1.0 - tf.div(uy, c_squ) + tf.multiply(tf.square(tf.div(uy, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  # next neighbour ones
  FEQ_1 = t3*density*(1.0 + tf.div(u_c2, c_squ) + tf.multiply(tf.square(tf.div(u_c2, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_3 = t3*density*(1.0 + tf.div(u_c4, c_squ) + tf.multiply(tf.square(tf.div(u_c4, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_5 = t3*density*(1.0 + tf.div(u_c6, c_squ) + tf.multiply(tf.square(tf.div(u_c6, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  FEQ_7 = t3*density*(1.0 + tf.div(u_c8, c_squ) + tf.multiply(tf.square(tf.div(u_c8, c_squ)),0.5)-tf.div(u_squ, 2.0*c_squ))
  # final one
  FEQ_8 = t1*density*(1.0 - tf.div(u_squ, 2.0*c_squ))
  # put them all together
  FEQ = tf.concat(axis=3, values=[FEQ_0, FEQ_1, FEQ_2, FEQ_3, FEQ_4, FEQ_5, FEQ_6, FEQ_7, FEQ_8])
  return FEQ

def _collision(f, feq, tau):
  omega = 1/tau
  return omega*feq+(1.0-omega)*f

def _combine_boundary(f, f_boundary, boundary_cutter_inv):
  return tf.multiply(f, boundary_cutter_inv) + f_boundary

def _f_to_density(f, boundary):
  density = tf.expand_dims(tf.reduce_sum(f, 3), 3)
  return density
 
def _f_to_u(f, u_kernel, density, boundary):
  u = tf.div(_simple_conv(f, u_kernel), density)
  return u

def f_to_u_full(f):
  u_kernel = _u_kernel_D2Q9()
  density = tf.expand_dims(tf.reduce_sum(f, 3), 3)
  u = tf.div(_simple_conv(f, u_kernel), density)
  return u

def zeros_f(shape, density=1.0, solver_type="D2Q9"):
  if solver_type == "D2Q9":
    f = np.zeros([1] + shape + [9], dtype=np.float32)
    f[:,:,:,0] = 1.0*density/9.0
    f[:,:,:,1] = 1.0*density/36.0
    f[:,:,:,2] = 1.0*density/9.0
    f[:,:,:,3] = 1.0*density/36.0
    f[:,:,:,4] = 1.0*density/9.0
    f[:,:,:,5] = 1.0*density/36.0
    f[:,:,:,6] = 1.0*density/9.0
    f[:,:,:,7] = 1.0*density/36.0
    f[:,:,:,8] = 4.0*density/9.0
  #f = tf.Variable(f)
  f = tf.constant(f)
  return f

def add_weights_f(f, density=1.0):
  weights = _weights_D2Q9()
  f = f + weights
  return f 

def sub_weights_f(f, density=1.0):
  weights = _weights_D2Q9()
  f = f - weights
  return f 

def make_u_input(shape, value=0.1):
  u = np.zeros((1,shape[0],1,1))
  l = shape[0] - 2
  for i in xrange(shape[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[0,i,0,0] = vx
  u = u.astype(np.float32)
  u[0,0:2,0,0] = 0.0
  u[0,-3:-1,0,0] = 0.0
  u = tf.constant(u)
  return u

def loss_divergence(f):
  u = f_to_u_full(f)
  div = divergence.spatial_divergence_2d(u)
  return tf.reduce_sum(div)

def lbm_step(f, boundary, u_in, density=1.0, tau=1.7):
  # creates step to run the lattice boltzmann solver
  transfer_kernel = _transfer_kernel_D2Q9()
  u_kernel = _u_kernel_D2Q9()
  boundary_kernel = _boundary_kernel_D2Q9()
  boundary_cutter, boundary_cutter_inv = _create_boundary_cutter(boundary)

  # set velcity in
  f_init_vel =  _set_velocity_boundary(f, u_in, density, pos="y_lower")

  # propagate
  f_propagate = _propagate(f_init_vel, transfer_kernel)

  # split between boundary and not boundary
  f_boundary = _bounce_back(f_propagate, boundary_cutter, boundary_kernel)

  # calc density and u
  density = _f_to_density(f_propagate, boundary)
  u = _f_to_u(f_propagate, u_kernel, density, boundary)
  u = tf.multiply(u, (1.0 - boundary))
  density = tf.multiply(density, (1.0 - boundary))

  # calc feq
  feq = _u_to_feq(u, density)

  # collision step
  f_collision = _collision(f_propagate, feq, tau)

  # add boundarys back in
  f_fin = _combine_boundary(f_collision, f_boundary, boundary_cutter_inv)

  # make computation step
  step = tf.group(
    f.assign(f_fin))

  return step, u, f

def lbm_seq(f, boundary, u_in, seq_length, init_density=1.0, tau=1.7):
  # this function is really just used for training
  transfer_kernel = _transfer_kernel_D2Q9()
  u_kernel = _u_kernel_D2Q9()
  boundary_kernel = _boundary_kernel_D2Q9()
  boundary_cutter, boundary_cutter_inv = _create_boundary_cutter(boundary)

  f_store = []
  for i in xrange(seq_length):
    # set velcity in
    f_init_vel =  _set_velocity_boundary(f, u_in, init_density, pos="y_lower")
    # propagate
    f_propagate = _propagate(f_init_vel, transfer_kernel)
    # split between boundary and not boundary
    f_boundary = _bounce_back(f_propagate, boundary_cutter, boundary_kernel)
    # calc density and u then kill the boundary pieces to avoid nans
    density = _f_to_density(f_propagate, boundary)
    u = _f_to_u(f_propagate, u_kernel, density, boundary)
    u = tf.multiply(u, (1.0 - boundary))
    density = tf.multiply(density, (1.0 - boundary))
    # calc feq
    feq = _u_to_feq(u, density)
    # collision step
    f_collision = _collision(f_propagate, feq, tau)
    # add boundarys back in
    f = _combine_boundary(f_collision, f_boundary, boundary_cutter_inv)
    f_store.append(f)

  return f_store



