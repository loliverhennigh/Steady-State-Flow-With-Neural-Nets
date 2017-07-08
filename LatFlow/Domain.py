
import numpy as np
import tensorflow as tf
import cv2
from utils import *
import time
from tqdm import *

import D2Q9
import D3Q15
import D3Q19

class Domain():
  def __init__(self,
               method,
               nu, 
               Ndim,
               boundary,
               dx=1.0,
               dt=1.0):

    if method == "D2Q9":
      self.Nneigh = 9
      self.Dim    = 2
      self.W      = tf.reshape(D2Q9.WEIGHTS, (self.Dim + 1)*[1] + [self.Nneigh])
      self.C      = tf.reshape(D2Q9.LVELOC, self.Dim*[1] + [self.Nneigh,3])
      self.Op     = tf.reshape(D2Q9.BOUNCE, self.Dim*[1] + [self.Nneigh,self.Nneigh])
      self.St     = D2Q9.STREAM

    if method == "D3Q15":
      self.Nneigh = 15
      self.Dim    = 3
      self.W      = tf.reshape(D3Q15.WEIGHTS, (self.Dim + 1)*[1] + [self.Nneigh])
      self.C      = tf.reshape(D3Q15.LVELOC, self.Dim*[1] + [self.Nneigh,3])
      self.Op     = tf.reshape(D3Q15.BOUNCE, self.Dim*[1] + [self.Nneigh,self.Nneigh])
      self.St     = D3Q15.STREAM

    if method == "D3Q19":
      self.Nneigh = 19
      self.Dim    = 3
      self.W      = tf.reshape(D3Q19.WEIGHTS, (self.Dim + 1)*[1] + [self.Nneigh])
      self.C      = tf.reshape(D3Q19.LVELOC, self.Dim*[1] + [self.Nneigh,3])
      self.Op     = tf.reshape(D3Q19.BOUNCE, self.Dim*[1] + [self.Nneigh,self.Nneigh])
      self.St     = D3Q19.STREAM

    if nu is not list:
      nu = [nu]
   
    self.time   = 0.0
    self.dt     = dt
    self.dx     = dx
    self.Cs     = dx/dt
    self.Step   = 1
    self.Sc     = 0.17
    self.Ndim   = Ndim
    self.Ncells = np.prod(np.array(Ndim))
    if type(boundary).__module__ == np.__name__:
      boundary = tf.constant(boundary)
    self.boundary = boundary
    self.Nlat   = int(boundary.get_shape()[0])

    self.Nl     = len(nu)
    self.tau    = []
    self.G      = []
    self.Gs     = []
    self.Rhoref = []
    self.Psi    = []
    self.Gmix   = 0.0
      
    self.F       = []
    self.Ftemp   = []
    self.Vel     = []
    self.BForce  = []
    self.Rho     = []
    self.IsSolid = []
  
    for i in xrange(len(nu)):
      self.tau.append(     3.0*nu[i]*self.dt/(self.dx*self.dx)+0.5)
      self.G.append(       0.0)
      self.Gs.append(      0.0)
      self.Rhoref.append(  200.0)
      self.Psi.append(     4.0)
      self.F.append(       tf.Variable(np.zeros([self.Nlat] + Ndim + [self.Nneigh], dtype=np.float32)))
      self.Ftemp.append(   tf.Variable(np.zeros([self.Nlat] + Ndim + [self.Nneigh], dtype=np.float32)))
      self.Vel.append(     tf.Variable(np.zeros([self.Nlat] + Ndim + [3], dtype=np.float32)))
      self.BForce.append(  tf.Variable(np.zeros([self.Nlat] + Ndim + [3], dtype=np.float32)))
      self.Rho.append(     tf.Variable(np.zeros([self.Nlat] + Ndim + [1], dtype=np.float32)))
      self.IsSolid.append( tf.Variable(np.zeros([self.Nlat] + Ndim + [1], dtype=np.float32)))

    self.EEk = tf.zeros(self.Dim*[1] + [self.Nneigh])
    for n in xrange(3):
      for m in xrange(3):
        if self.Dim == 2:
          self.EEk = self.EEk + tf.abs(self.C[:,:,:,n] * self.C[:,:,:,m])
        elif self.Dim == 3:
          self.EEk = self.EEk + tf.abs(self.C[:,:,:,:,n] * self.C[:,:,:,:,m])

  def CollideSC(self, graph_unroll=False):
    # boundary bounce piece
    f_boundary = tf.multiply(self.F[0], self.boundary)
    f_boundary = simple_conv(f_boundary, self.Op)

    # make vel bforce and rho
    f   = self.F[0]
    vel = self.Vel[0]
    rho = self.Rho[0] + 1e-10 # to stop dividing by zero

    # calc v dots
    #vel = vel_no_boundary + self.dt*self.tau[0]*(bforce_no_boundary/(rho_no_boundary + 1e-10))
    vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=self.Dim+1), axis=self.Dim+1)
    if self.Dim == 2:
      vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0,1,3,2]))
    else:
      vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0,1,2,4,3]))

    # calc Feq
    Feq = self.W * rho * (1.0 + 3.0*vel_dot_c/self.Cs + 4.5*vel_dot_c*vel_dot_c/(self.Cs*self.Cs) - 1.5*vel_dot_vel/(self.Cs*self.Cs))

    # collision calc
    NonEq = f - Feq
    Q = tf.expand_dims(tf.reduce_sum(NonEq*NonEq*self.EEk, axis=self.Dim+1), axis=self.Dim+1)
    Q = tf.sqrt(1e-6 + 2.0*Q)
    tau = 0.5*(self.tau[0]+tf.sqrt(self.tau[0]*self.tau[0] + 6.0*Q*self.Sc/rho))
    f = f - NonEq/tau

    # combine boundary and no boundary values
    f_no_boundary = tf.multiply(f, (1.0-self.boundary))
    f = f_no_boundary + f_boundary

    if not graph_unroll:
      # make step
      collid_step = self.F[0].assign(f)
      return collid_step
    else:
      # put computation back in graph
      self.F[0] = f
      

  """
  def CollideMP(self):
    # boundary bounce piece
    f_boundary = tf.multiply(self.F[0], self.boundary)
    f_boundary = simple_conv(f_boundary, self.Op)

    # make vel_mix, rho_mix, vel, and rho
    f   = self.F
    vel_mix = tf.expand_dims(tf.reduce_sum(self.Vel*self.Rho/self.tau, axis=0), axis=0)
    rho_mix = tf.expand_dims(tf.reduce_sum(self.Rho/self.tau + 1e-10 , axis=0), axis=0) # to stop dividing by zero
    vel_mix = vel_mix/rho_mix
    rho = self.Rho
    vel = vel_mix
    
    # calc v dots
    #vel = vel_no_boundary + self.dt*self.tau[0]*(bforce_no_boundary/(rho_no_boundary + 1e-10))
    vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=self.Dim+1), axis=self.Dim+1)
    if self.Dim == 2:
      vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0,1,3,2]))
    else:
      vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0,1,2,4,3]))

    # calc Feq
    Feq = self.W * rho * (1.0 + 3.0*vel_dot_c/self.Cs + 4.5*vel_dot_c*vel_dot_c/(self.Cs*self.Cs) - 1.5*vel_dot_vel/(self.Cs*self.Cs))

    # collision calc
    f = f - (f - Feq)/self.tau

    # combine boundary and no boundary values
    f_no_boundary = tf.multiply(f, (1.0-self.boundary))
    f = f_no_boundary + f_boundary
    collid_step = self.F.assign(f)
    return collid_step
  """

  def StreamSC(self, graph_unroll=False):
    # stream f
    f_pad = pad_mobius(self.F[0])
    f_pad = simple_conv(f_pad, self.St)
    # calc new velocity and density
    Rho = tf.expand_dims(tf.reduce_sum(f_pad, self.Dim+1), self.Dim+1)
    Vel = simple_conv(f_pad, self.C)
    Vel = Vel/(self.Cs * Rho)
    if not graph_unroll:
      # create steps
      stream_step = self.F[0].assign(f_pad)
      Rho_step =    self.Rho[0].assign(Rho)
      Vel_step =    self.Vel[0].assign(Vel)
      step = tf.group(*[stream_step, Rho_step, Vel_step])
      return step
    else:
      self.F[0] = f_pad
      self.Rho[0] = Rho
      self.Vel[0] = Vel

  """
  def StreamMP(self):
    # stream f
    f_pad = pad_mobius(self.F)
    f_pad = simple_conv(f_pad, self.St)
    # calc new velocity and density
    Rho = tf.expand_dims(tf.reduce_sum(f_pad, self.Dim+1), self.Dim+1)
    Vel = simple_conv(f_pad, self.C)
    Vel = Vel/(self.Cs * Rho)
    # create steps
    stream_step = self.F.assign(f_pad)
    Rho_step =    self.Rho.assign(Rho)
    Vel_step =    self.Vel.assign(Vel)
    step = tf.group(*[stream_step, Rho_step, Vel_step])
    return step
  """

  def Initialize(self, graph_unroll=False):
    np_f_zeros = np.zeros([1] + self.Ndim + [self.Nneigh], dtype=np.float32)
    f_zero = tf.constant(np_f_zeros)
    f_zero = f_zero + self.W
    if not graph_unroll:
      assign_step = self.F[0].assign(f_zero)
      return assign_step 
    else:
      self.F[0] = f_zero

  def Solve(self, sess, Tf, initialize_step, setup_step, save_step, save_interval):
    # make steps
    assign_step = self.Initialize()
    stream_step = self.StreamSC() 
    collide_step = self.CollideSC()

    # run solver
    sess.run(assign_step)
    sess.run(initialize_step)
    num_steps = int(Tf/self.dt)
    for i in tqdm(xrange(num_steps)):
      if int(self.time/save_interval) > int((self.time-self.dt)/save_interval):
        save_step(self, sess)
      sess.run(setup_step) 
      sess.run(collide_step)
      sess.run(stream_step)
      self.time += self.dt

  def Unroll(self, start_f, num_steps, setup_computation, setup_params):
    # run solver
    self.F[0] = start_f
    self.StreamSC(graph_unroll=True)
    F_return_state = []
    for i in xrange(num_steps):
      setup_computation(self, *setup_params)
      self.CollideSC(graph_unroll=True)
      self.StreamSC(graph_unroll=True)
      F_return_state.append(self.F[0])
    return F_return_state




