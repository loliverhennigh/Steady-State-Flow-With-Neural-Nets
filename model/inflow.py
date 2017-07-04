
import tensorflow as tf
import numpy as np
from LatFlow.utils import *

def get_inflow_vector(name="parabolic"):
  inflow_vector = None  
  if name == "parabolic":
    inflow_vector = parabolic_flow_vector
  elif name == "uniform":
    inflow_vector = uniform_flow_vector
  return inflow_vector

def uniform_flow_vector(shape, value, pos="y_0"):
  u     = np.zeros([1] + shape + [3])
  u_on  = np.zeros([1] + shape + [1])
  value = np.array(value).reshape((len(shape))*[1] + [3])
  if pos == "y_0":
    u[:,:,0]     = u[:,:,0] + value
    u_on[:,:,0]  = 1.0
  elif pos == "y_n":
    u[:,:,-1]    = u[:,:,-1] + value
    u_on[:,:,-1] = 1.0
  elif pos == "x_0":
    u[:,0]       = u[:,0] + value
    u_on[:,0]    = 1.0
  elif pos == "x_n":
    u[:,-1]      = u[:,-1] + value
    u_on[:,-1]   = 1.0
  elif pos == "z_0":
    u[:,:,:,0]       = u[:,:,:,0] + value
    u_on[:,:,:,0]    = 1.0
  elif pos == "z_n":
    u[:,:,:,-1]      = u[:,:,:,-1] + value
    u_on[:,:,:,-1]   = 1.0
  u    = u.astype(np.float32)
  u_on = u_on.astype(np.float32)
  return u, u_on

def parabolic_flow_vector(shape, value, pos="x_0"):
  # only 2D with x_0 inlet implemented (unsure of how to do 3D)
  u     = np.zeros([1] + shape + [3])
  u_on  = np.zeros([1] + shape + [1])
  value = np.array(value).reshape((len(shape))*[1] + [3])
  l = shape[0] - 2
  for i in xrange(shape[0]):
    yp = i - 1.5
    vx = value*4.0/(l*l)*(l*yp - yp*yp)
    u[:,i,0]    = vx
    u_on[:,i,0] = 1.0
  u    = u.astype(np.float32)
  u_on = u_on.astype(np.float32)
  return u, u_on

def apply_flow(domain, u, u_on):
  ndim = len(u.get_shape())-2

  vel = u
  vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=ndim+1), axis=ndim+1)
  vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=ndim+1) * domain.C, axis=ndim+2)
  feq = domain.W * (1.0 + (3.0/domain.Cs)*vel_dot_c + (4.5/(domain.Cs*domain.Cs))*(vel_dot_c*vel_dot_c) - (1.5/(domain.Cs*domain.Cs))*vel_dot_vel)

  vel = vel * (1.0 - domain.boundary)
  rho = (1.0 - domain.boundary)

  #f   = (domain.F[0]  *(1.0 - u_on)) + (domain.F[0]  *(1.0 - u_on))(feq*u_on)
  f   = (domain.F[0]  *(1.0 - u_on)) + (feq*u_on)
  vel = (domain.Vel[0]*(1.0 - u_on)) + (vel*u_on)
  rho = (domain.Rho[0]*(1.0 - u_on)) + (rho*u_on)

  #domain.F[0]   = feq
  domain.F[0]   = f
  domain.Rho[0] = rho
  domain.Vel[0] = vel


