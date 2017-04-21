
"""functions used to construct different architectures  
"""


import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def int_shape(x):
  return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))

def set_nonlinearity(name):
  if name == 'concat_elu':
    return concat_elu
  elif name == 'elu':
    return tf.nn.elu
  elif name == 'concat_relu':
    return tf.nn.crelu
  elif name == 'relu':
    return tf.nn.relu
  else:
    raise('nonlinearity ' + name + ' is not supported')

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable(name, shape, initializer):
  """Helper to create a Variable.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  # getting rid of stddev for xavier ## testing this for faster convergence
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = int(inputs.get_shape()[3])

    weights = _variable('weights', shape=[kernel_size,kernel_size,input_channels,num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)
    return conv_biased

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = int(inputs.get_shape()[3])
    
    weights = _variable('weights', shape=[kernel_size,kernel_size,num_features,input_channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)

    #reshape
    shape = int_shape(inputs)
    conv_biased = tf.reshape(conv_biased, [shape[0], shape[1]*stride, shape[2]*stride, num_features])

    return conv_biased

def fc_layer(inputs, hiddens, idx, nonlinearity=None, flat = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable('weights', shape=[dim,hiddens],initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [hiddens], initializer=tf.contrib.layers.xavier_initializer())
    output_biased = tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
    if nonlinearity is not None:
      output_biased = nonlinearity(ouput_biased)
    return output_biased

def nin(x, num_units, idx):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = fc_layer(x, num_units, idx)
    return tf.reshape(x, s[:-1]+[num_units])

def _phase_shift(I, r):
  bsize, a, b, c = I.get_shape().as_list()
  bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
  X = tf.reshape(I, (bsize, a, b, r, r))
  X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
  X = tf.split(axis=1, num_or_size_splits=a, value=X)  # a, [bsize, b, r, r]
  X = tf.concat(axis=2, values=[tf.squeeze(x) for x in X])  # bsize, b, a*r, r
  X = tf.split(axis=1, num_or_size_splits=b, value=X)  # b, [bsize, a*r, r]
  X = tf.concat(axis=2, values=[tf.squeeze(x) for x in X])  # bsize, a*r, b*r
  return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, depth):
  Xc = tf.split(axis=3, num_or_size_splits=depth, value=X)
  X = tf.concat(axis=3, values=[_phase_shift(x, r) for x in Xc])
  return X

def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False, name="resnet"):
  orig_x = x
  orig_x_int_shape = int_shape(x)
  if orig_x_int_shape[3] == 1:
    x_1 = conv_layer(x, 3, stride, filter_size, name + '_conv_1')
  else:
    x_1 = conv_layer(nonlinearity(x), 3, stride, filter_size, name + '_conv_1')
  if a is not None:
    shape_a = int_shape(a) 
    shape_x_1 = int_shape(x_1)
    a = tf.pad(
      a, [[0, 0], [0, shape_x_1[1]-shape_a[1]], [0, shape_x_1[2]-shape_a[2]],
      [0, 0]])
    x_1 += nin(nonlinearity(a), filter_size, name + '_nin')
  x_1 = nonlinearity(x_1)
  if keep_p < 1.0:
    x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)
  if not gated:
    x_2 = conv_layer(x_1, 3, 1, filter_size, name + '_conv_2')
  else:
    x_2 = conv_layer(x_1, 3, 1, filter_size*2, name + '_conv_2')
    x_2_1, x_2_2 = tf.split(axis=3,num_or_size_splits=2,value=x_2)
    x_2 = x_2_1 * tf.nn.sigmoid(x_2_2)

  if int(orig_x.get_shape()[2]) > int(x_2.get_shape()[2]):
    assert(int(orig_x.get_shape()[2]) == 2*int(x_2.get_shape()[2]), "res net block only supports stirde 2")
    orig_x = tf.nn.avg_pool(orig_x, [1,2,2,1], [1,2,2,1], padding='SAME')

  # pad it
  out_filter = filter_size
  in_filter = int(orig_x.get_shape()[3])
  if out_filter != in_filter:
    orig_x = tf.pad(
        orig_x, [[0, 0], [0, 0], [0, 0],
        [(out_filter-in_filter), 0]])

  return orig_x + x_2

def conv_res(inputs, nr_res_blocks=1, keep_prob=1.0, nonlinearity_name='concat_elu', gated=True):
  """Builds conv part of net.
  Args:
    inputs: input images
    keep_prob: dropout layer
  """
  nonlinearity = set_nonlinearity(nonlinearity_name)
  filter_size = 8
  # store for as
  a = []
  # res_1
  x = inputs
  for i in xrange(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_1_" + str(i))
  # res_2
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_2_downsample")
  for i in xrange(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_2_" + str(i))
  # res_3
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_3_downsample")
  for i in xrange(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_3_" + str(i))
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_4_downsample")
  for i in xrange(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_4_" + str(i))
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_5_downsample")
  for i in xrange(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_5_" + str(i))
  # res_up_1
  filter_size = filter_size /2
  x = transpose_conv_layer(x, 3, 2, filter_size, "up_conv_1")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-1], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i))
    else:
      x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i))
  # res_up_1
  filter_size = filter_size /2
  x = transpose_conv_layer(x, 3, 2, filter_size, "up_conv_2")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-2], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i))
    else:
      x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i))

  filter_size = filter_size /2
  x = transpose_conv_layer(x, 3, 2, filter_size, "up_conv_3")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-3], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i))
    else:
      x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i))
 
  filter_size = filter_size /2
  x = transpose_conv_layer(x, 3, 2, filter_size, "up_conv_4")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-4], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i))
    else:
      x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i))
  
  x = conv_layer(x, 3, 1, 2, "last_conv")
  x = tf.nn.tanh(x) 

  tf.summary.image('sflow_p_x', x[:,:,:,1:2])
  tf.summary.image('sflow_p_v', x[:,:,:,0:1])

  return x


