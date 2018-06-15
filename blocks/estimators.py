import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.helpers import int_shape, log_sum_exp

def compute_kernel(x, y, is_dimention_wise=False):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    if is_dimention_wise:
        return tf.exp(-tf.square(tiled_x - tiled_y))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_gaussian_entropy(z_mu, z_log_sigma_sq):
    batch_size, z_dim = int_shape(z_mu)
    entropy = (tf.reduce_mean(z_log_sigma_sq, axis=1) + tf.log(2*np.pi*np.e)) * z_dim / 2.
    return entropy


def compute_gaussian_kld(z_mu, z_log_sigma_sq, output_mean=True):
    kld = - 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq), axis=-1)
    if output_mean:
        kld = tf.reduce_mean(kld)
    return kld


def estimate_mmd(x, y, is_dimention_wise=False):
    x_kernel = compute_kernel(x, x, is_dimention_wise)
    y_kernel = compute_kernel(y, y, is_dimention_wise)
    xy_kernel = compute_kernel(x, y, is_dimention_wise)
    if is_dimention_wise:
        mmd = tf.reduce_mean(x_kernel, axis=[0,1]) + tf.reduce_mean(y_kernel, axis=[0,1]) - 2 * tf.reduce_mean(xy_kernel, axis=[0,1])
        return tf.reduce_sum(mmd)
    mmd = tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    return mmd

def estimate_mmdtc(y, indices):
    batch_size, z_dim = int_shape(y)
    batch_size = batch_size // 2
    x, y = y[:batch_size], y[batch_size:]
    ys = tf.unstack(y, axis=1)
    for i in range(z_dim):
        ys[i] = tf.gather(ys[i], indices[:, i])
    y = tf.stack(ys, axis=1)
    return estimate_mmd(x, y)

def estimate_log_probs(z, z_mu, z_log_sigma_sq, N=200000):
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    log_probs = []
    batch_size, z_dim = int_shape(z_mu)

    z_b = tf.stack([z for i in range(batch_size)], axis=0)
    z_mu_b = tf.stack([z_mu for i in range(batch_size)], axis=1)
    z_sigma_b = tf.stack([z_sigma for i in range(batch_size)], axis=1)
    z_norm = (z_b-z_mu_b) / z_sigma_b

    dist = tf.distributions.Normal(loc=0., scale=1.)
    log_probs = dist.log_prob(z_norm)
    ratio = np.log(float(N-1) / (batch_size-1)) * np.ones((batch_size, batch_size))
    np.fill_diagonal(ratio, 0.)
    ratio_b = np.stack([ratio for i in range(z_dim)], axis=-1)

    lse_sum = tf.reduce_mean(log_sum_exp(tf.reduce_sum(log_probs, axis=-1)+ratio, axis=0))
    sum_lse = tf.reduce_mean(tf.reduce_sum(log_sum_exp(log_probs+ratio_b, axis=0), axis=-1))
    return lse_sum, sum_lse

def estimate_tc(z, z_mu, z_log_sigma_sq, N=200000):
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    log_probs = []
    batch_size, z_dim = int_shape(z_mu)

    z_b = tf.stack([z for i in range(batch_size)], axis=0)
    z_mu_b = tf.stack([z_mu for i in range(batch_size)], axis=1)
    z_sigma_b = tf.stack([z_sigma for i in range(batch_size)], axis=1)
    z_norm = (z_b-z_mu_b) / z_sigma_b

    dist = tf.distributions.Normal(loc=0., scale=1.)
    log_probs = dist.log_prob(z_norm)
    ratio = np.log(float(N-1) / (batch_size-1)) * np.ones((batch_size, batch_size))
    np.fill_diagonal(ratio, 0.)
    ratio_b = np.stack([ratio for i in range(z_dim)], axis=-1)

    lse_sum = tf.reduce_mean(log_sum_exp(tf.reduce_sum(log_probs, axis=-1)+ratio, axis=0))
    sum_lse = tf.reduce_mean(tf.reduce_sum(log_sum_exp(log_probs+ratio_b, axis=0), axis=-1))
    return lse_sum - sum_lse + tf.log(float(N)) * (float(z_dim)-1)


def estimate_dwkld(z, z_mu, z_log_sigma_sq, N=200000):
    batch_size, z_dim = int_shape(z_mu)
    lse_sum, sum_lse = estimate_log_probs(z, z_mu, z_log_sigma_sq, N=N)
    sum_lse -= tf.log(float(N)) * float(z_dim)

    kld = compute_gaussian_kld(z_mu, z_log_sigma_sq)
    cond_entropy = tf.reduce_mean(compute_gaussian_entropy(z_mu, z_log_sigma_sq))
    nll_prior = kld + cond_entropy

    return sum_lse + nll_prior

def estimate_mi(z, z_mu, z_log_sigma_sq, N=200000):
    batch_size, z_dim = int_shape(z_mu)
    lse_sum, sum_lse = estimate_log_probs(z, z_mu, z_log_sigma_sq, N=N)
    lse_sum -= tf.log(float(N))
    cond_entropy = tf.reduce_mean(compute_gaussian_entropy(z_mu, z_log_sigma_sq))
    return -lse_sum - cond_entropy

def estimate_mi_tc_dwkld(z, z_mu, z_log_sigma_sq, N=2e5):
    # computational cheaper to compute them together
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    log_probs = []
    batch_size, z_dim = int_shape(z_mu)

    z_b = tf.stack([z for i in range(batch_size)], axis=0)
    z_mu_b = tf.stack([z_mu for i in range(batch_size)], axis=1)
    z_sigma_b = tf.stack([z_sigma for i in range(batch_size)], axis=1)
    z_norm = (z_b-z_mu_b) / z_sigma_b

    dist = tf.distributions.Normal(loc=0., scale=1.)
    log_probs = dist.log_prob(z_norm)
    ratio = np.log(float(N-1) / (batch_size-1)) * np.ones((batch_size, batch_size))
    np.fill_diagonal(ratio, 0.)
    ratio_b = np.stack([ratio for i in range(z_dim)], axis=-1)

    lse_sum = tf.reduce_mean(log_sum_exp(tf.reduce_sum(log_probs, axis=-1)+ratio, axis=0))
    sum_lse = tf.reduce_mean(tf.reduce_sum(log_sum_exp(log_probs+ratio_b, axis=0), axis=-1))
    lse_sum -= tf.log(float(N))
    sum_lse -= tf.log(float(N)) * float(z_dim)

    kld = compute_gaussian_kld(z_mu, z_log_sigma_sq)
    cond_entropy = tf.reduce_mean(compute_gaussian_entropy(z_mu, z_log_sigma_sq))

    mi = -lse_sum - cond_entropy
    tc = lse_sum - sum_lse
    dwkld = sum_lse + (kld + cond_entropy)

    return mi, tc, dwkld
