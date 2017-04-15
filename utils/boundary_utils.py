
import numpy as np

#import matplotlib.pyplot as plt


def fill_bottom_flat_triangle(boundary, vertex_1, vertex_2, vertex_3):
  inv_slope_1 = (vertex_2[0] - vertex_1[0]) / (vertex_2[1] - vertex_1[1])
  inv_slope_2 = (vertex_3[0] - vertex_1[0]) / (vertex_3[1] - vertex_1[1])

  cur_x_1 = vertex_1[0]
  cur_x_2 = vertex_1[0]

  for i in xrange(int(vertex_1[1]),  int(vertex_2[1]), 1):
    max_cur = max(cur_x_1, cur_x_2)
    min_cur = min(cur_x_1, cur_x_2)
    boundary[int(min_cur-1.5):int(max_cur+1.5),i] = 1.0
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
    boundary[int(min_cur-1.5):int(max_cur+1.5),i] = 1.0
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
    vertex_4[1] = vertex_2[1] - 1
    fill_bottom_flat_triangle(boundary, vertex_1, vertex_2, vertex_4)
    fill_top_flat_triangle(boundary, vertex_2, vertex_4, vertex_3)

  return boundary

def sort_vertices(vertex_1, vertex_2, vertex_3):
  data = [vertex_1, vertex_2, vertex_3]
  data.sort(key=lambda tup: tup[1])
  return data[0], data[1], data[2]

def make_boundary_circle(length_input, shape):
  boundary = np.zeros(shape)
  max_length = np.min(shape)/4.0
  pos = np.zeros((2))
  pos[0] = int(shape[0]/2.0)
  pos[1] = int(shape[1]/2.0)

  x_1 = np.zeros((2))
  x_1[0] = 0 + pos[0]
  x_1[1] = int(length_input[0]*max_length) + pos[1]
  x_start = np.copy(x_1)

  x_2 = np.zeros((2))

  alpha = (2*np.pi)/len(length_input)
  alpha_i = 0.0

  for i in xrange(len(length_input)-1):

    length = length_input[i+1]

    alpha_i += alpha

    x_2[0] = int(np.sin(alpha_i)*length*max_length + pos[0])
    x_2[1] = int(np.cos(alpha_i)*length*max_length + pos[1])
    
    draw_triangle(boundary, pos, x_1, x_2) 
 
    x_1 = np.copy(x_2)

  draw_triangle(boundary, pos, x_start, x_2) 

  return boundary

#length_input = np.zeros((10)) + 1.0
#length_input = np.random.rand(1009)
#boundary = make_boundary_circle(length_input, (10000,10000))

#plt.figure()
#plt.imshow(boundary)
#plt.show()



