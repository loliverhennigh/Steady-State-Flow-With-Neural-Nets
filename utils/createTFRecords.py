
import numpy as np 
import tensorflow as tf 
from glob import glob as glb
import re
from tqdm import *
from flow_reader import load_flow, load_boundary, load_state

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_bool('debug', False,
                            """ this will show the images while generating records. """)

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# create tf writer
record_filename = '../data/train.tfrecords'

writer = tf.python_io.TFRecordWriter(record_filename)

# the stored frames
shape = [128, 256]
frames = np.zeros((shape[0], shape[1], 1))

# list of files
train_filename = glb('/data/fluid_flow_steady_state_128x128_/*') 

for run in tqdm(train_filename):
  # read in images
  flow_name = run + '/fluid_flow_0002.h5'
  boundary = load_boundary(flow_name, shape)
  sflow = load_state(flow_name, shape)
  
  # Display the resulting frame
  if FLAGS.debug == True:
    cv2.imshow('boundary', boundary) 
    cv2.waitKey(0)
    cv2.imshow('sflow', sflow[:,:,0]) 
    cv2.waitKey(0)
   
  # process frame for saving
  boundary = np.uint8(boundary)
  boundary = boundary.reshape([1,shape[0]*shape[1]])
  boundary = boundary.tostring()
  sflow = np.float32(sflow)
  sflow = sflow.reshape([1,shape[0]*shape[1]*9])
  sflow = sflow.tostring()
  
  # create example and write it
  example = tf.train.Example(features=tf.train.Features(feature={
    'boundary': _bytes_feature(boundary),
    'sflow': _bytes_feature(sflow)})) 
  writer.write(example.SerializeToString()) 


