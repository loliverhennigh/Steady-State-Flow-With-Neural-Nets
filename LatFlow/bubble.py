
import time

import numpy as np
import tensorflow as tf
import math 
import cv2
from utils import *

import Domain as dom

import matplotlib.pyplot as plt 

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

shape = [64, 64, 64]

success = video.open('video.mov', fourcc, 60, (128, 128), True)

FLAGS = tf.app.flags.FLAGS

def make_bubble_boundary(shape):
  boundary = np.zeros([1] + shape + [1], dtype=np.float32)
  return boundary

def run():
  # constants
  nu = 1.0/6.0
  Ndim = shape
  boundary = make_bubble_boundary(shape=Ndim)

  # domain
  domain = dom.Domain("D3Q15", nu, Ndim, boundary)
  domain

  # make lattice state, boundary and input velocity
  initialize_step = ball_init_step(domain, value=0.08)
  setup_step = ball_setup_step(domain, value=input_vel)

  # init things
  init = tf.global_variables_initializer()

  # start sess
  sess = tf.Session()

  # init variables
  sess.run(init)

  # run steps
  domain.Solve(sess, 100, initialize_step, setup_step, video)

def main(argv=None):  # pylint: disable=unused-argument
  run()

if __name__ == '__main__':
  tf.app.run()




