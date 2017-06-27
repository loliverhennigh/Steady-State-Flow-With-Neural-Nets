
import tensorflow as tf

def simple_conv(x, k):
  """A simplified 2D or 3D convolution operation"""
  if   len(x.get_shape()) == 4:
    y = tf.nn.conv2d(x, k, [1, 1, 1, 1],    padding='VALID')
  elif len(x.get_shape()) == 5:
    y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='VALID')
  return y

def pad_mobius(f):
  f_mobius = f
  f_mobius = tf.concat(axis=1, values=[f_mobius[:,-1:],   f_mobius, f_mobius[:,0:1]]) 
  f_mobius = tf.concat(axis=2, values=[f_mobius[:,:,-1:], f_mobius, f_mobius[:,:,0:1]])
  if len(f.get_shape()) == 5:
    f_mobius = tf.concat(axis=3, values=[f_mobius[:,:,:,-1:], f_mobius, f_mobius[:,:,:,0:1]])
  return f_mobius
 
