
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_boundary_creator(name):
  boundary_creator = None
  if name == 'oval':
    boundary_creator = make_rand_boundary_circle
  elif name == 'shapes':
    boundary_creator = make_rand_boundary
  return boundary_creator

def draw_triangle(boundary, vertex_1, vertex_2, vertex_3):
  # just using cv2 imp
  triangle = np.array([[vertex_1[1],vertex_1[0]],[vertex_2[1],vertex_2[0]],[vertex_3[1],vertex_3[0]]], np.int32)
  triangle = triangle.reshape((-1,1,2))
  cv2.fillConvexPoly(boundary,triangle,1)
  return boundary

def draw_random_triangle(boundary, x_range, y_range, size_range):
  size_x_1 = np.random.randint(-size_range, size_range)
  size_y_1 = np.random.randint(-size_range, size_range)
  size_x_2 = np.random.randint(-size_range, size_range)
  size_y_2 = np.random.randint(-size_range, size_range)
  max_length_x = np.max([np.abs(size_x_1), np.abs(size_x_2)])
  max_length_y = np.max([np.abs(size_y_1), np.abs(size_y_2)])
  vertex = rand_vertex([max_length_x, shape[0]-max_length_x], [max_length_y+int(.1*shape[1]), shape[1]-max_length_y-int(.1*shape[1])])
  boundary = draw_triangle(boundary, vertex, [vertex[0]+size_x_2, vertex[1]+size_y_2], [vertex[0]+size_x_1, vertex[1]+size_y_1])
  return boundary

def draw_ovel(boundary, vertex, cord_1, cord_2, angle, nr_angles=20):
  alpha = (2*np.pi)/nr_angles
  alpha_i = 0.0
  degree_i = angle
  x_1 = np.zeros((2))
  length = np.square(np.sin(degree_i))*cord_1 + np.square(np.cos(degree_i))*cord_2
  x_1[0] = int(np.sin(alpha_i)*length + vertex[0])
  x_1[1] = int(np.cos(alpha_i)*length + vertex[1])
  x_2 = np.copy(x_1)
  x_start = np.copy(x_1)
  for i in xrange(nr_angles):
    length = np.square(np.sin(degree_i)*cord_1)/cord_1 + np.square(np.cos(degree_i)*cord_2)/cord_2
    x_2[0] = int(np.sin(alpha_i)*length + vertex[0])
    x_2[1] = int(np.cos(alpha_i)*length + vertex[1])
    alpha_i -= alpha
    degree_i -= alpha
    draw_triangle(boundary, vertex, x_1, x_2)
    x_1 = np.copy(x_2)
  draw_triangle(boundary, vertex, x_start, x_2)
  return boundary

def slice_length(length_input, index, degree):
  length_out = np.zeros(degree)
  for i in xrange(degree):
    length_out[i] = length_input[(index + i - (degree-1)/2) % len(length_input)]
  return length_out

def get_length(length_input, index, degree, rate_curvy):
  length_out = slice_length(length_input, index, degree)
  total_length = 0.0
 
  # determine constant b of parabola
  b = -rate_curvy*pow(degree-1., 2)/(degree-1.)
  sum_left = 0.0
  for i in xrange(degree):
    sum_left += rate_curvy*pow(i,2) + b*i
  c =  (1.0-sum_left)/degree
 
  sum_constant = 0.0
  for i in xrange(degree):
    constant = rate_curvy*pow(i,2) + b*i + c
    total_length += constant * length_out[i]
    sum_constant += constant
  return total_length

def rand_vertex(range_x, range_y):
  pos_x = np.random.randint(range_x[0], range_x[1])
  pos_y = np.random.randint(range_y[0], range_y[1])
  vertex = np.array([pos_x, pos_y])
  return vertex

def make_boundary_circle(length_input, shape, degree=9, rate_curvy=-.01):
  boundary = np.zeros(shape)
  max_length = np.min(shape)/4.0
  pos = np.zeros((2))
  pos[0] = int(shape[0]/2.0)
  pos[1] = int(shape[1]/2.0)

  x_1 = np.zeros((2))
  x_1[0] = 0 + pos[0]
  x_1[1] = int(max_length*get_length(length_input, 0, degree, rate_curvy)) + pos[1]
  x_start = np.copy(x_1)
  x_2 = np.zeros((2))
  alpha = (2*np.pi)/len(length_input)
  alpha_i = 0.0
  for i in xrange(len(length_input-1)):
    length = get_length(length_input, i+1, degree, rate_curvy)
    x_2[0] = int(np.sin(alpha_i)*length*max_length + pos[0])
    x_2[1] = int(np.cos(alpha_i)*length*max_length + pos[1])
    alpha_i -= alpha
    draw_triangle(boundary, pos, x_1, x_2)
    x_1 = np.copy(x_2)
  draw_triangle(boundary, pos, x_start, x_2)
  return boundary

def make_rand_boundary_circle(shape):
  rand_length = np.random.rand(31)
  boundary = make_boundary_circle(rand_length, shape)
  boundary[0:1,:] = 1
  boundary[-1:,:] = 1
  return boundary

#def make_rand_boundary(shape, num_objects_range=[2,12], size_range=[10,30], max_boundary=2500):
def make_rand_boundary(shape, num_objects_range=[1,4], size_range=[.05,.15], max_boundary=.25):
  shape_max = np.min(shape)
  size_range = [int(size_range[0]*shape_max), int(size_range[1]*shape_max)]
  boundary = np.zeros(shape)
  max_boundary = max_boundary * shape[0] * shape[1]
  num_objects = np.random.randint(num_objects_range[0], num_objects_range[1])
  for i in xrange(num_objects):
    object_type = np.random.randint(0,2)
    # draw circle
    if object_type == 0:
      size_x = np.random.randint(size_range[0], size_range[1])
      size_y = np.random.randint(size_range[0], size_range[1])
      angle = np.random.randint(0, 90)
      max_length = np.max([size_x, size_y])
      vertex = rand_vertex([max_length, shape[0]-max_length], [max_length+int(.1*shape[1]), shape[1]-max_length-int(.1*shape[1])])
      boundary = draw_ovel(boundary, vertex, size_x, size_y, angle, nr_angles=20)
    if object_type == 1:
      size_x_1 = np.random.randint(-size_range[1], size_range[1])
      size_y_1 = np.random.randint(-size_range[1], size_range[1])
      size_x_2 = np.random.randint(-size_range[1], size_range[1])
      size_y_2 = np.random.randint(-size_range[1], size_range[1])
      max_length_x = np.max([np.abs(size_x_1), np.abs(size_x_2)])
      max_length_y = np.max([np.abs(size_y_1), np.abs(size_y_2)])
      vertex = rand_vertex([max_length_x, shape[0]-max_length_x], [max_length_y+int(.1*shape[1]), shape[1]-max_length_y-int(.1*shape[1])])
      boundary = draw_triangle(boundary, vertex, [vertex[0]+size_x_2, vertex[1]+size_y_2], [vertex[0]+size_x_1, vertex[1]+size_y_1])
    if np.sum(boundary) > max_boundary:
      break
  boundary[0:1,:] = 1
  boundary[-1:,:] = 1
  boundary = np.expand_dims(boundary, axis=0)
  boundary = np.expand_dims(boundary, axis=3)
  return boundary

"""
for i in xrange(10):     
  length_input = np.random.rand(30)
  #boundary = make_boundary_circle(length_input, [256,512])
  boundary = make_rand_boundary([256,512])
  plt.figure()
  plt.imshow(boundary)
  plt.show()
"""



