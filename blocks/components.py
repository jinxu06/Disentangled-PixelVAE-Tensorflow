import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.helpers import int_shape, get_name, broadcast_masks_tf
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shift, right_shift


# # big conv blocks
# @add_arg_scope
# def conv_encoder_64_large(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
#     name = get_name("conv_encoder_64_large", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
#             outputs = inputs
#             outputs = conv2d(outputs, 32, 1, 1, "SAME")
#             outputs = conv2d(outputs, 64, 4, 2, "SAME")
#             outputs = conv2d(outputs, 128, 4, 2, "SAME")
#             outputs = conv2d(outputs, 256, 4, 2, "SAME")
#             outputs = conv2d(outputs, 512, 4, 2, "SAME")
#             outputs = conv2d(outputs, 1024, 4, 1, "VALID")
#             outputs = tf.reshape(outputs, [-1, 1024])
#             z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
#             z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
#             return z_mu, z_log_sigma_sq
#
#
# @add_arg_scope
# def conv_decoder_64_large(inputs, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
#     name = get_name("conv_decoder_64_large", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
#             outputs = dense(inputs, 1024)
#             outputs = tf.reshape(outputs, [-1, 1, 1, 1024])
#             outputs = deconv2d(outputs, 512, 4, 1, "VALID")
#             outputs = deconv2d(outputs, 256, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 128, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 64, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 32, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 3, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=False)
#             outputs = 2. * outputs - 1.
#             return outputs


# medium
@add_arg_scope
def conv_encoder_64_medium(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_64_medium", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 2, "SAME")
            outputs = conv2d(outputs, 512, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 512])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq

@add_arg_scope
def conv_decoder_64_medium(inputs, is_training, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_decoder_64_medium", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            if output_features:
                return deconv2d(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d(outputs, 3, 4, 2, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
            return outputs


# # small
# @add_arg_scope
# def conv_encoder_64_small(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
#     name = get_name("conv_encoder_64_small", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
#             outputs = inputs
#             outputs = conv2d(outputs, 32, 4, 2, "SAME")
#             outputs = conv2d(outputs, 32, 4, 2, "SAME")
#             outputs = conv2d(outputs, 64, 4, 2, "SAME")
#             outputs = conv2d(outputs, 64, 4, 2, "SAME")
#             outputs = tf.reshape(outputs, [-1, 64*4*4])
#             outputs = dense(outputs, 256, bn=False)
#             z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
#             z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
#             return z_mu, z_log_sigma_sq
#
#
#
#
# @add_arg_scope
# def conv_decoder_64_small(inputs, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
#     name = get_name("conv_decoder_64_small", counters)
#     print("construct", name, "...")
#     with tf.variable_scope(name):
#         with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
#             outputs = dense(inputs, 256, bn=False)
#             outputs = dense(outputs, 4*4*64, nonlinearity=tf.nn.tanh, bn=False)
#             outputs = tf.reshape(outputs, [-1, 4, 4, 64])
#             outputs = deconv2d(outputs, 64, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 32, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 32, 4, 2, "SAME")
#             outputs = deconv2d(outputs, 3, 4, 2, "SAME", nonlinearity=tf.sigmoid, bn=False)
#             outputs = 2. * outputs - 1.
#             return outputs

@add_arg_scope
def conv_encoder_32_large1(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_32_large1", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d(outputs, 32, 1, 1, "SAME")
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 2, "SAME")
            outputs = conv2d(outputs, 512, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 512])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq

@add_arg_scope
def conv_decoder_32_large1(inputs, is_training, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_decoder_32_large1", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d(outputs, 32, 4, 2, "SAME")
            if output_features:
                return outputs
            outputs = deconv2d(outputs, 3, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
            return outputs

@add_arg_scope
def conv_encoder_32_large(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_32_large", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d(outputs, 32, 1, 1, "SAME")
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 2, "SAME")
            outputs = conv2d(outputs, 512, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 512])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=True)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=True)
            return z_mu, z_log_sigma_sq


@add_arg_scope
def conv_decoder_32_large(inputs, is_training, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_decoder_32_large", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d(outputs, 32, 4, 2, "SAME")
            if output_features:
                return outputs
            outputs = deconv2d(outputs, 3, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=True)
            outputs = 2. * outputs - 1.
            return outputs


@add_arg_scope
def conv_encoder_32_medium(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_32_medium", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 256])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq

@add_arg_scope
def conv_decoder_32_medium(inputs, is_training, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_decoder_32_medium", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense(inputs, 256)
            outputs = tf.reshape(outputs, [-1, 1, 1, 256])
            outputs = deconv2d(outputs, 128, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            if output_features:
                return deconv2d(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d(outputs, 3, 4, 2, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
            return outputs



@add_arg_scope
# nr_filters= 32
def context_encoder(contexts, masks, is_training, nr_resnet=5, nr_filters=100, nonlinearity=None, bn=False, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("context_encoder", counters)
    print("construct", name, "...")
    x = contexts * broadcast_masks_tf(masks, num_channels=3)
    x = tf.concat([x, broadcast_masks_tf(masks, num_channels=1)], axis=-1)
    if bn:
        print("*** Attention *** using bn in the context encoder\n")
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], nonlinearity=nonlinearity, counters=counters):
            with arg_scope([gated_resnet, up_shifted_conv2d, up_left_shifted_conv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [up_shift(up_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [up_shift(up_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        left_shift(up_left_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                receptive_field = (2, 3)
                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=up_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=up_left_shifted_conv2d))
                    receptive_field = (receptive_field[0]+1, receptive_field[1]+2)
                x_out = nin(tf.nn.elu(ul_list[-1]), nr_filters)
                print("    * receptive_field", receptive_field)
                return x_out


@add_arg_scope
def cond_pixel_cnn(x, gh=None, sh=None, nonlinearity=tf.nn.elu, nr_resnet=5, nr_filters=100, nr_logistic_mix=10, bn=False, dropout_p=0.0, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_pixel_cnn", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=gh, sh=sh, nonlinearity=nonlinearity, dropout_p=dropout_p, counters=counters):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                receptive_field = (2, 3)
                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))
                    receptive_field = (receptive_field[0]+1, receptive_field[1]+2)
                x_out = nin(tf.nn.elu(ul_list[-1]), 10*nr_logistic_mix)
                print("    * receptive_field", receptive_field)
                return x_out
