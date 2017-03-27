
import os
import numpy as np
import tensorflow as tf
from glob import glob as glb


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_integer('min_queue_examples', 1000,
                           """ min examples to queue up""")

def read_data(filename_queue, shape):
  """ reads data from tfrecord files.

  Args: 
    filename_queue: A que of strings with filenames 
    shape: image shape 

  Returns:
    frames: the frame data in size (batch_size, image height, image width, frames)
  """
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'boundary':tf.FixedLenFeature([],tf.string),
      'sflow':tf.FixedLenFeature([],tf.string)
    }) 
  boundary = tf.decode_raw(features['boundary'], tf.uint8)
  sflow = tf.decode_raw(features['sflow'], tf.float32)
  boundary = tf.reshape(boundary, [shape[0], shape[1], 1])
  sflow = tf.reshape(sflow, [shape[0], shape[1], 2])
  boundary = tf.to_float(boundary)
  sflow = tf.to_float(sflow) 
  return boundary, sflow 

def _generate_image_label_batch(boundary, sflow, batch_size, shuffle=True):
  """Construct a queued batch of images.
  Args:
    image: 3-D Tensor of [height, width, frame_num] 
    mask: 3-D Tensor of [height, width, frame_num] 
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
  """

  num_preprocess_threads = 1
  #Create a queue that shuffles the examples, and then
  #read 'batch_size' images + labels from the example queue.
  boundarys, sflows = tf.train.shuffle_batch(
    [boundary, sflow],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=FLAGS.min_queue_examples + 3 * batch_size,
    min_after_dequeue=FLAGS.min_queue_examples)
  return boundarys, sflows

def flow_inputs(batch_size):
  """ Construct nerve input net.
  Args:
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
    mask: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
  """

  shape = (128,256)

  tfrecord_filename = glb('../data/*') 
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  boundary, sflow = read_data(filename_queue, shape)

  boundarys, sflows = _generate_image_label_batch(boundary, sflow, batch_size)
 
  # display in tf summary page 
  tf.summary.image('boundarys', boundarys)
  tf.summary.image('sflows_x', sflows[:,:,:,1:2])
  tf.summary.image('sflows_y', sflows[:,:,:,0:1])

  return boundarys, sflows 

