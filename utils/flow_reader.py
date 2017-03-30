
import h5py
import numpy as np 

WEIGHTD2Q9 = np.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.])
LVELOCD2Q9 = np.array([ [0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1] ])

def subtract_lattice(lattice, Weights):
  # this increases percesion before converting to 32 bit
  Weights = Weights.reshape((1,1,Weights.shape[0]))
  lattice = lattice - Weights
  return lattice

def load_flow(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_boundary(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def load_state(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state = np.array(stream_flow['State_0'][:])
  flow_state = flow_state.reshape([shape[0], shape[1]+128, 9])[0:shape[0],0:shape[1],:]
  flow_state = subtract_lattice(flow_state, WEIGHTD2Q9)
  stream_flow.close()
  return flow_state

