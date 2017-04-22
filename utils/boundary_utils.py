
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
    boundary[int(np.floor(min_cur)):int(np.floor(max_cur)),i] = 1.0
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
    boundary[int(np.floor(min_cur)):int(np.floor(max_cur)),i] = 1.0
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

def sort_vertices(vertex_1, vertex_2, vertex_3):
  data = [vertex_1, vertex_2, vertex_3]
  data.sort(key=lambda tup: tup[1])
  return data[0], data[1], data[2]

def slice_length(length_input, index, degree):
  length_out = np.zeros(degree*2)
  for i in xrange(degree*2):
    length_out[i] = length_input[index + i - degree]
  return length_out

def get_length(length_input, index, degree, rate_curvey):
  length_out = slice_length(length_input, index, degree)
  total_length = 0.0
 
  # determine constant b of parabola
  b = -rate_curvey*pow(degree-1, 2)
  sum_left = 1.0
  sum_right = 0.0
  for i in xrange(degree):
    sum_left -= rate_curvey*pow(i,2)
    sum_right += i
  b = sum_left/sum_right
 
  sum_constant = 0.0
  for i in xrange(degree):
    constant = rate_curvey*pow(i,2) + b*i
    print(constant)
    total_length += constant * length_out[i]
    sum_constant += constant
  print(sum_constant)
  print(total_length)
  return total_length

def make_boundary_circle(length_input, shape, degree_curvey=5, rate_curvey=-.4):
  boundary = np.zeros(shape)
  max_length = np.min(shape)/2.0
  pos = np.zeros((2))
  pos[0] = int(shape[0]/2.0)
  pos[1] = int(shape[1]/2.0)

  x_1 = np.zeros((2))
  x_1[0] = 0 + pos[0]
  x_1[1] = get_length(length_input, 0, degree_curvey, rate_curvey) + pos[1]
  x_start = np.copy(x_1)

  x_2 = np.zeros((2))

  alpha = (2*np.pi)/len(length_input)
  alpha_i = 0.0

  for i in xrange(len(length_input)):

    length = get_length(length_input, i, degree_curvey, rate_curvey)

    alpha_i += alpha

    x_2[0] = int(np.sin(alpha_i)*length*max_length + pos[0])
    x_2[1] = int(np.cos(alpha_i)*length*max_length + pos[1])
    
    draw_triangle(boundary, pos, x_1, x_2) 
 
    x_1 = np.copy(x_2)

  draw_triangle(boundary, pos, x_start, x_2) 

  return boundary

def make_boundary_curvey(length_input, shape):
  boundary = np.zeros(shape)
  max_length = np.min(shape)/2.0

  pos = np.zeros((2))
  pos[0] = int(shape[0]/2.0)
  pos[1] = int(shape[1]/2.0)

  alpha = (2*np.pi)/len(length_input)
  alpha_i = 0.0

  for i in xrange(shape[0]):
    for j in xrange(shape[1]):
      angle = (i-pos[0]) (i-pos[0])
      pos 
      boundary[i,j]


#length_input = np.zeros((19)) + 0.5
length_input = np.random.rand(29)
boundary = make_boundary_circle(length_input, (128,128))

plt.figure()
plt.imshow(boundary)
plt.show()



