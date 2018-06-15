import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.helpers import int_shape, get_name

# @add_arg_scope
# def conv2d(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
#     outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#     if nonlinearity is not None:
#         outputs = nonlinearity(outputs)
#     if bn:
#         outputs = tf.layers.batch_normalization(outputs, training=is_training)
#     print("    + conv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
#     return outputs
#
# @add_arg_scope
# def deconv2d(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
#     outputs = tf.layers.conv2d_transpose(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#     if nonlinearity is not None:
#         outputs = nonlinearity(outputs)
#     if bn:
#         outputs = tf.layers.batch_normalization(outputs, training=is_training)
#     print("    + deconv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
#     return outputs
#
# @add_arg_scope
# def dense(inputs, num_outputs, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
#     inputs_shape = int_shape(inputs)
#     assert len(inputs_shape)==2, "inputs should be flattened first"
#     outputs = tf.layers.dense(inputs, num_outputs, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#     if nonlinearity is not None:
#         outputs = nonlinearity(outputs)
#     if bn:
#         outputs = tf.layers.batch_normalization(outputs, training=is_training)
#     print("    + dense", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
#     return outputs

@add_arg_scope
def conv2d(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + conv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
    return outputs

@add_arg_scope
def deconv2d(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    outputs = tf.layers.conv2d_transpose(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + deconv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
    return outputs

@add_arg_scope
def dense(inputs, num_outputs, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    inputs_shape = int_shape(inputs)
    assert len(inputs_shape)==2, "inputs should be flattened first"
    outputs = tf.layers.dense(inputs, num_outputs, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + dense", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
    return outputs



def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

def up_shift(x):
    xs = int_shape(x)
    return tf.concat([x[:,1:xs[1],:,:], tf.zeros([xs[0],1,xs[2],xs[3]])],1)

def left_shift(x):
    xs = int_shape(x)
    return tf.concat([x[:,:,1:xs[2],:], tf.zeros([xs[0],xs[1],1,xs[3]])],2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

# @add_arg_scope
# def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], strides=[1,1], **kwargs):
#     x = deconv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)
#     xs = int_shape(x)
#     return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

# @add_arg_scope
# def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], strides=[1,1], **kwargs):
#     x = deconv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)
#     xs = int_shape(x)
#     return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]

@add_arg_scope
def up_shifted_conv2d(x, num_filters, filter_size=[2,3], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[0, filter_size[0]-1], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def up_left_shifted_conv2d(x, num_filters, filter_size=[2,2], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[0, filter_size[0]-1], [0, filter_size[1]-1],[0,0]])
    return conv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)


@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])

@add_arg_scope
def gated_resnet(x, a=None, gh=None, sh=None, nonlinearity=tf.nn.elu, conv=conv2d, dropout_p=0.0, counters={}, **kwargs):
    name = get_name("gated_resnet", counters)
    print("construct", name, "...")
    xs = int_shape(x)
    num_filters = xs[-1]
    with arg_scope([conv], **kwargs):
        c1 = conv(nonlinearity(x), num_filters)
        if a is not None: # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), num_filters)
        c1 = nonlinearity(c1)
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = conv(c1, num_filters * 2)
        # add projection of h vector if included: conditional generation
        if sh is not None:
            c2 += nin(sh, 2*num_filters, nonlinearity=nonlinearity)
        if gh is not None: # haven't finished this part
            pass
        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)
        return x + c3
