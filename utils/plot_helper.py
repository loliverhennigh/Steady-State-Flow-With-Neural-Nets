
import numpy as np

def grey_to_short_rainbow(grey):
  max_grey = np.max(grey)
  grey = grey/max_grey
  a = (1-grey)/0.25
  x = np.floor(a)
  y = np.floor(255*(a-x))
  rainbow = np.zeros((grey.shape[0], grey.shape[1], 3))
  for i in xrange(x.shape[0]):
    for j in xrange(x.shape[1]):
      if x[i,j,0] == 0:
        rainbow[i,j,2] = 255
        rainbow[i,j,1] = y[i,j,0]
        rainbow[i,j,0] = 0
      if x[i,j,0] == 1:
        rainbow[i,j,2] = 255 - y[i,j,0]
        rainbow[i,j,1] = 255
        rainbow[i,j,0] = 0
      if x[i,j,0] == 2:
        rainbow[i,j,2] = 0
        rainbow[i,j,1] = 255
        rainbow[i,j,0] = y[i,j,0] 
      if x[i,j,0] == 3:
        rainbow[i,j,2] = 0
        rainbow[i,j,1] = 255 - y[i,j,0]
        rainbow[i,j,0] = 255
      if x[i,j,0] == 4:
        rainbow[i,j,2] = 0
        rainbow[i,j,1] = 0
        rainbow[i,j,0] = 255
  return rainbow
