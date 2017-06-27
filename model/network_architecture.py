
import tensorflow as tf
import numpy as np
import nn
import LatFlow.D2Q9  as D2Q9
import LatFlow.D3Q15 as D3Q15

def residual_u_network(inputs, density=1.0, start_filter_size=16, nr_downsamples=4, nr_residual_per_downsample=2, keep_prob=1.0, nonlinearity="concat_elu"):

  # set filter size (after each down sample the filter size is doubled)
  filter_size = start_filter_size
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  # store for u network connections
  a = []
  # encoding piece
  x_i = inputs
  print(x_i.get_shape())
  for i in xrange(nr_downsamples):
    x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_0", begin_nonlinearity=False)
    for j in xrange(nr_residual_per_downsample - 1):
      x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
    a.append(x_i)
    filter_size = filter_size * 2
  # pop off last element to a.
  a.pop()
  filter_size = filter_size / 2
  # decoding piece
  for i in xrange(nr_downsamples - 1):
    filter_size = filter_size / 2
    #x_i = nn.upsampleing_resize(x_i, filter_size, "up_conv_" + str(i))
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    x_i = nn.res_block(x_i, a=a.pop(), filter_size=filter_size, keep_p=keep_prob, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_0", begin_nonlinearity=True)
    for j in xrange(nr_residual_per_downsample-1):
      x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
  x_i = nn.transpose_conv_layer(x_i, 2, 2, 9, "up_conv_" + str(nr_downsamples-1))
  #x_i = nn.upsampleing_resize(x_i, 9, "up_conv_" + str(nr_downsamples-1))
  # create into flow dist
  x_i = tf.nn.tanh(x_i)
  #x_i = .9 * lb.mul_weights_f(x_i)
  #x_i = lb.add_weights_f(x_i, density=density)
  return x_i

def conv_res(inputs, nr_res_blocks=1, keep_prob=1.0, nonlinearity_name='concat_elu', gated=True):
  """Builds conv part of net.
  Args:
    inputs: input images
    keep_prob: dropout layer
  """
  nonlinearity = nn.set_nonlinearity(nonlinearity_name)
  filter_size = 8
  # store for as
  a = []
  # res_1
  x = inputs
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_1_" + str(i))
  # res_2
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_2_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_2_" + str(i))
  # res_3
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_3_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_3_" + str(i))
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_4_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_4_" + str(i))
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_5_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_5_" + str(i))
  # res_up_1
  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_1")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-1], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i))
  # res_up_1
  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_2")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-2], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i))

  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_3")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-3], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i))
 
  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_4")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-4], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i))
  
  x = nn.conv_layer(x, 3, 1, 9, "last_conv")
  x = tf.nn.tanh(x) 
  x = .9 * tf.reshape(D2Q9.WEIGHTS, [1,1,1,9]) * x
  x = tf.reshape(D2Q9.WEIGHTS, [1,1,1,9]) + x
  print("is running!!!!!!!!!!")

  tf.summary.image('sflow_p_x', x[:,:,:,1:2])
  tf.summary.image('sflow_p_v', x[:,:,:,0:1])

  return x

def fc_conv(inputs, shape, nonlinearity_name="elu"):
  nonlinearity = nn.set_nonlinearity(nonlinearity_name)
  fc_1 = nn.fc_layer(inputs, shape[0]*shape[1]/8, 0, nonlinearity=nonlinearity)
  fc_1 = tf.reshape(fc_1, [-1, shape[0]/16, shape[0]/8, 32])
  fconv_1 = nn.transpose_conv_layer(fc_1, 3, 2, 32, "up_conv_1", nonlinearity=nonlinearity)
  fconv_2 = nn.transpose_conv_layer(fconv_1, 3, 2, 16, "up_conv_2", nonlinearity=nonlinearity)
  fconv_3 = nn.transpose_conv_layer(fconv_2, 3, 2, 8, "up_conv_3", nonlinearity=nonlinearity)
  boundary = nn.transpose_conv_layer(fconv_3, 3, 2, 1, "up_conv_4", nonlinearity=nonlinearity)
  boundary = nn.tf.sigmoid(boundary)
  return boundary
 



