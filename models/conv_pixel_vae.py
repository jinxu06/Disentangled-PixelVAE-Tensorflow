import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense
from blocks.samplers import gaussian_sampler, mix_logistic_sampler
from blocks.estimators import estimate_mi_tc_dwkld, estimate_mmd, compute_gaussian_kld
from blocks.losses import mix_logistic_loss
from blocks.helpers import int_shape, broadcast_masks_tf
from blocks.components import conv_encoder_64_medium, conv_decoder_64_medium, conv_encoder_32_medium, conv_decoder_32_medium, conv_encoder_32_large, conv_decoder_32_large, conv_encoder_32_large1, conv_decoder_32_large1
from blocks.components import cond_pixel_cnn, context_encoder


class ConvPixelVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, x, x_bar, is_training, dropout_p, z_dim, masks=None, input_masks=None, use_mode="test", network_size="medium", reg='mmd', N=2e5, sample_range=1.0, beta=1., lam=0., nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None, nr_resnet=1, nr_filters=100, nr_logistic_mix=10):
        self.z_dim = z_dim
        self.use_mode = use_mode
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.reg = reg
        self.N = N
        self.sample_range = sample_range
        self.beta = beta
        self.lam = lam
        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
        self.nr_logistic_mix = nr_logistic_mix
        self.__model(x, x_bar, is_training, dropout_p, masks, input_masks, network_size=network_size)
        self.__loss(self.reg)

    def __model(self, x, x_bar, is_training, dropout_p, masks, input_masks, network_size="medium"):
        print("******   Building Graph   ******")
        self.x = x
        self.x_bar = x_bar
        self.is_training = is_training
        self.dropout_p = dropout_p
        self.masks = masks
        self.input_masks = input_masks
        if int_shape(x)[1]==64:
            conv_encoder = conv_encoder_64_medium
            conv_decoder = conv_decoder_64_medium
        elif int_shape(x)[1]==32:
            if network_size=='medium':
                conv_encoder = conv_encoder_32_medium
                conv_decoder = conv_decoder_32_medium
            elif network_size=='large':
                conv_encoder = conv_encoder_32_large
                conv_decoder = conv_decoder_32_large
            elif network_size=='large1':
                conv_encoder = conv_encoder_32_large1
                conv_decoder = conv_decoder_32_large1
            else:
                raise Exception("unknown network type")
        with arg_scope([conv_encoder, conv_decoder, context_encoder, cond_pixel_cnn], nonlinearity=self.nonlinearity, bn=self.bn, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, is_training=self.is_training, counters=self.counters):
            inputs = self.x
            if self.input_masks is not None:
                inputs = inputs * broadcast_masks_tf(self.input_masks, num_channels=3)
                inputs = tf.concat([inputs, broadcast_masks_tf(self.input_masks, num_channels=1)], axis=-1)
            self.z_mu, self.z_log_sigma_sq = conv_encoder(inputs, self.z_dim)
            sigma = tf.exp(self.z_log_sigma_sq / 2.)
            if self.use_mode=='train':
                self.z = gaussian_sampler(self.z_mu, sigma)
            elif self.use_mode=='test':
                self.z = tf.placeholder(tf.float32, shape=int_shape(self.z_mu))
            print("use mode:{0}".format(self.use_mode))
            self.decoded_features = conv_decoder(self.z, output_features=True)
            if self.masks is None:
                sh = self.decoded_features
            else:
                self.encoded_context = context_encoder(self.x, self.masks, bn=False, nr_resnet=self.nr_resnet, nr_filters=self.nr_filters)
                sh = tf.concat([self.decoded_features, self.encoded_context], axis=-1)
            self.mix_logistic_params = cond_pixel_cnn(self.x_bar, sh=sh, bn=False, dropout_p=self.dropout_p, nr_resnet=self.nr_resnet, nr_filters=self.nr_filters, nr_logistic_mix=self.nr_logistic_mix)
            self.x_hat = mix_logistic_sampler(self.mix_logistic_params, nr_logistic_mix=self.nr_logistic_mix, sample_range=self.sample_range, counters=self.counters)



    def __loss(self, reg):
        print("******   Compute Loss   ******")
        self.mmd, self.kld, self.mi, self.tc, self.dwkld = [None for i in range(5)]
        self.gamma, self.dwmmd = 1e3, None ## hard coded, experimental
        self.loss_ae = mix_logistic_loss(self.x, self.mix_logistic_params, masks=self.masks)
        if reg is None:
            self.loss_reg = 0
        elif reg=='kld':
            self.kld = compute_gaussian_kld(self.z_mu, self.z_log_sigma_sq)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.kld)
        elif reg=='mmd':
            # self.mmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z)
            self.mmd = estimate_mmd(tf.random_normal(tf.stack([256, self.z_dim])), self.z)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.mmd)
        elif reg=='tc':
            self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
            self.loss_reg = self.mi + self.beta * self.tc + self.dwkld
        elif reg=='tc-dwmmd':
            self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
            self.dwmmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z, is_dimention_wise=True)
            self.loss_reg = self.beta * self.tc + self.dwmmd * self.gamma
        elif reg=='mmd-tc':
            self.mmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z)
            self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
            self.loss_reg = self.beta * self.tc + self.mmd * 1e5

        print("reg:{0}, beta:{1}, lam:{2}".format(self.reg, self.beta, self.lam))
        self.loss = self.loss_ae + self.loss_reg
