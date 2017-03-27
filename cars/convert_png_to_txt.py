
#from __future__ import print_function

import numpy as np
import cv2
from matplotlib import pyplot as plt

save_shape = (300, 100)


i = 1
while True:
  # load png file
  img = cv2.imread('car_' + str(i).zfill(3) + '.png', 0)
  img = cv2.flip(img, 1)
  if img is None:
    break
  resized_img = cv2.resize(img, save_shape)
  resized_img = -(resized_img/255.0) + 1

  # write to txt file
  f1=open('car_' + str(i).zfill(3) + '.txt', 'w+')
  print >>f1, str(save_shape[0]) + ' ',
  print >>f1, str(save_shape[1]) + ' ',
  for x in xrange(save_shape[0]):
    print >>f1, '\n',
    for y in xrange(save_shape[1]):
      if resized_img[y,x] > .5:
        print >>f1, str(1) + ' ',
      if resized_img[y,x] < .5:
        print >>f1, str(0) + ' ',

  i += 1



