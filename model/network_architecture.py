
import tensorflow as tf
import numpy as np
import lb_solver as lb
import nn

def residual_u_network(inputs, start_filter_size=16, nr_downsamples=4, nr_residual_per_downsample=2, nonlinearity="concat_elu"):

  # set filter size (after each down sample the filter size is doubled)
  filter_size = start_filter_size
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  # store for u network connections
  a = []
  # encoding piece
  x_i = inputs
  for i in xrange(nr_downsamples):
    x_i = nn.res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_0", begin_nonlinearity=False)
    for j in xrange(nr_residual_per_downsample - 1):
      x_i = nn.res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
    a.append(x_i)
    filter_size = filter_size * 2
  # pop off last element to a.
  a.pop()
  filter_size = filter_size / 2
  # decoding piece
  for i in xrange(nr_downsamples - 1):
    filter_size = filter_size / 2
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual_per_downsample):
      x_i = nn.res_block(x_i, a=a.pop(), filter_size=filter_size, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
  x_i = nn.transpose_conv_layer(x_i, 9, 2, int(inputs.get_shape()[-1]), "up_conv_" + str(nr_downsamples-1))
  return x_i

def fc_conv(inputs, nonlinearity_name="elu"):
  nonlinearity = nn.set_nonlinearity(nonlinearity_name)
  fc_1 = nn.fc_layer(inputs, 2048, 0, nonlinearity)
  fc_1 = tf.reshape(fc_1, [-1, 8, 16, 16])
  fconv_1 = nn.transpose_conv_layer(fc_1, 3, 2, 16, "up_conv_1")
  fconv_2 = nn.transpose_conv_layer(fconv_1, 3, 2, 8, "up_conv_2")
  fconv_3 = nn.transpose_conv_layer(fconv_2, 3, 2, 4, "up_conv_3")
  boundary = nn.transpose_conv_layer(fconv_3, 3, 2, 1, "up_conv_4")
  boundary = nn.tf.sigmoid(boundary)
  return boundary
 



