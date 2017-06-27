
import tensorflow as tf
import numpy as np
from LatFlow.utils import *

def get_inflow(name="parabolic"):
  inflow = None  
  if name == "parabolic":
    inflow = parabolic_flow
  elif name == "uniform":
    inflow = uniform_flow
  return inflow

def get_inflow_vector(name="parabolic"):
  inflow_vector = None  
  if name == "parabolic":
    inflow_vector = parabolic_flow_vector
  elif name == "uniform":
    inflow_vector = uniform_flow_vector
  return inflow_vector

def uniform_flow_vector(shape, value=0.1):
  u = np.zeros(shape)
  u[:,0] = u[:,0] + value
  u = u.astype(np.float32)
  return u

def parabolic_flow_vector(shape, value=0.1):
  u = np.zeros(shape)
  l = shape[0] - 2
  for i in xrange(shape[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[i,0] = vx
  u = u.astype(np.float32)
  return u

def uniform_flow(domain, value=0.1):
  u = np.zeros((1,domain.Ndim[0],1,1)) + value
  u = u.astype(np.float32)
  u = tf.constant(u)

  # input vel on left side
  f_out = domain.F[0][:,:,1:]
  f_edge = tf.split(domain.F[0][:,:,0:1], 9, axis=3)

  # new in distrobution
  rho = (f_edge[0] + f_edge[2] + f_edge[4] + 2.0*(f_edge[3] + f_edge[6] + f_edge[7]))/(1.0 - u)
  f_edge[1] = f_edge[3] + (2.0/3.0)*rho*u
  f_edge[5] = f_edge[7] + (1.0/6.0)*rho*u - 0.5*(f_edge[2]-f_edge[4])
  f_edge[8] = f_edge[6] + (1.0/6.0)*rho*u + 0.5*(f_edge[2]-f_edge[4])
  f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
  f = tf.concat([f_edge,f_out],axis=2)
 
  # new Rho
  rho = domain.Rho[0]
  rho_out = rho[:,:,1:]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_edge,rho_out],axis=2)

  # new vel
  vel = domain.Vel[0]
  vel_out = vel[:,:,1:]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_edge,vel_out],axis=2)

  # remove vel on right side
  f_out = f[:,:,:-1]
  f_edge = tf.split(f[:,:,-1:], 9, axis=3)

  # new out distrobution
  vx = -1.0 + (f_edge[0] + f_edge[2] + f_edge[4] + 2.0*(f_edge[1] + f_edge[5] + f_edge[8]))
  f_edge[3] = f_edge[1] - (2.0/3.0)*vx
  f_edge[7] = f_edge[5] - (1.0/6.0)*vx + 0.5*(f_edge[2]-f_edge[4])
  f_edge[6] = f_edge[8] - (1.0/6.0)*vx - 0.5*(f_edge[2]-f_edge[4])
  f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
  f = tf.concat([f_out,f_edge],axis=2)
 
  # new Rho
  rho_out = rho[:,:,:-1]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_out,rho_edge],axis=2)

  # new vel
  vel_out = vel[:,:,:-1]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_out,vel_edge],axis=2)

  # make steps
  domain.F[0] = f
  domain.Rho[0] = rho
  domain.Vel[0] = vel

def parabolic_flow(domain, value=0.001):
  u = np.zeros((1,domain.Ndim[0],1,1))
  l = domain.Ndim[0] - 2
  for i in xrange(domain.Ndim[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[0,i,0,0] = vx
  u = u.astype(np.float32)
  u = tf.constant(u)

  # input vel on left side
  f_out = domain.F[0][:,:,1:]
  f_edge = tf.split(domain.F[0][:,:,0:1], 9, axis=3)

  # new in distrobution
  rho = (f_edge[0] + f_edge[2] + f_edge[4] + 2.0*(f_edge[3] + f_edge[6] + f_edge[7]))/(1.0 - u)
  f_edge[1] = f_edge[3] + (2.0/3.0)*rho*u
  f_edge[5] = f_edge[7] + (1.0/6.0)*rho*u - 0.5*(f_edge[2]-f_edge[4])
  f_edge[8] = f_edge[6] + (1.0/6.0)*rho*u + 0.5*(f_edge[2]-f_edge[4])
  f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
  f = tf.concat([f_edge,f_out],axis=2)
 
  # new Rho
  rho = domain.Rho[0]
  rho_out = rho[:,:,1:]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_edge,rho_out],axis=2)

  # new vel
  vel = domain.Vel[0]
  vel_out = vel[:,:,1:]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_edge,vel_out],axis=2)

  # remove vel on right side
  f_out = f[:,:,:-1]
  f_edge = tf.split(f[:,:,-1:], 9, axis=3)

  # new out distrobution
  vx = -1.0 + (f_edge[0] + f_edge[2] + f_edge[4] + 2.0*(f_edge[1] + f_edge[5] + f_edge[8]))
  f_edge[3] = f_edge[1] - (2.0/3.0)*vx
  f_edge[7] = f_edge[5] - (1.0/6.0)*vx + 0.5*(f_edge[2]-f_edge[4])
  f_edge[6] = f_edge[8] - (1.0/6.0)*vx - 0.5*(f_edge[2]-f_edge[4])
  f_edge = tf.stack(f_edge, axis=3)[:,:,:,:,0]
  f = tf.concat([f_out,f_edge],axis=2)
 
  # new Rho
  rho_out = rho[:,:,:-1]
  rho_edge = tf.expand_dims(tf.reduce_sum(f_edge, axis=3), axis=3)
  rho = tf.concat([rho_out,rho_edge],axis=2)

  # new vel
  vel_out = vel[:,:,:-1]
  vel_edge = simple_conv(f_edge, tf.reshape(domain.C, [1,1,domain.Nneigh, 3]))
  vel_edge = vel_edge/rho_edge
  vel = tf.concat([vel_out,vel_edge],axis=2)

  # make steps
  domain.F[0] = f
  domain.Rho[0] = rho
  domain.Vel[0] = vel
