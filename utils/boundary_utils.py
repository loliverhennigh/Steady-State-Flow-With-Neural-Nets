
import numpy as np

import matplotlib.pyplot as plt

def fill_bottom_flat_triangle(boundary, vertex_1, vertex_2, vertex_3):
  inv_slope_1 = (vertex_2[0] - vertex_1[0]) / (vertex_2[1] - vertex_1[1])
  inv_slope_2 = (vertex_3[0] - vertex_1[0]) / (vertex_3[1] - vertex_1[1])

  cur_x_1 = vertex_1[0]
  cur_x_2 = vertex_1[0]

  for i in xrange(int(vertex_1[1]),  int(vertex_2[1]), 1):
    max_cur = max(cur_x_1, cur_x_2)
    min_cur = min(cur_x_1, cur_x_2)
    #boundary[int(min_cur-1.5):int(max_cur+1.5),i] = 1.0
    boundary[int(np.floor(min_cur-.2)):int(np.floor(max_cur+.2)),i] = 1.0
    cur_x_1 += inv_slope_1
    cur_x_2 += inv_slope_2

  return boundary

def fill_top_flat_triangle(boundary, vertex_1, vertex_2, vertex_3):
  inv_slope_1 = (vertex_3[0] - vertex_1[0]) / (vertex_3[1] - vertex_1[1])
  inv_slope_2 = (vertex_3[0] - vertex_2[0]) / (vertex_3[1] - vertex_2[1])

  cur_x_1 = vertex_3[0]
  cur_x_2 = vertex_3[0]

  for i in xrange(int(vertex_3[1]),  int(vertex_2[1]), -1):
    max_cur = max(cur_x_1, cur_x_2)
    min_cur = min(cur_x_1, cur_x_2)
    #boundary[int(min_cur-1.5):int(max_cur+1.5),i] = 1.0
    boundary[int(np.floor(min_cur-.2)):int(np.floor(max_cur+.2)),i] = 1.0
    cur_x_1 -= inv_slope_1
    cur_x_2 -= inv_slope_2

  return boundary

def draw_triangle(boundary, vertex_1, vertex_2, vertex_3):
  vertex_1, vertex_2, vertex_3 = sort_vertices(vertex_1, vertex_2, vertex_3)

  if vertex_2[1] == vertex_3[1]:
    fill_bottom_flat_triangle(boundary, vertex_1, vertex_2, vertex_3)
  elif vertex_1[1] == vertex_2[1]:
    fill_top_flat_triangle(boundary, vertex_1, vertex_2, vertex_3)
  else:
    vertex_4 = np.zeros((2))
    vertex_4[0] = int(np.floor(vertex_1[0] + ((vertex_2[1] - vertex_1[1]) / (vertex_3[1] - vertex_1[1])) * (vertex_3[0] - vertex_1[0])))
    vertex_4[1] = vertex_2[1] - 0.5
    #vertex_4[1] = vertex_2[1] 
    fill_bottom_flat_triangle(boundary, vertex_1, vertex_2, vertex_4)
    fill_top_flat_triangle(boundary, vertex_2, vertex_4, vertex_3)

  return boundary

def draw_rectangle(boundary, vertex_1, vertex_2, vertex_3):
  # not implemented
  boundary = draw_triangle(boundary, vertex_1, vertex_2, vertex_3)
  vertex_4 = 
  
def draw_ovel(boundary, vertex_1, cord_1, cord_2, angle, nr_angles=20):
  for i in xrange(nr_angles):
    x_length = cos(

def sort_vertices(vertex_1, vertex_2, vertex_3):
  data = [vertex_1, vertex_2, vertex_3]
  data.sort(key=lambda tup: tup[1])
  return data[0], data[1], data[2]

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

def make_boundary_circle(length_input, shape, degree=9, rate_curvy=-.01):
  boundary = np.zeros(shape)
  max_length = np.min(shape)/2.0
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

length_input = np.random.rand(39)
boundary = make_boundary_circle(length_input, (128,128))

#plt.figure()
#plt.imshow(boundary)
#plt.show()



