import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from PIL import Image

def int_shape(x):
    return list(map(int, x.get_shape()))

def log_sum_exp(x, axis=-1):
    return tf.reduce_logsumexp(x, axis=axis)

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def PIL_to_uint(pil_img):
    pass

def uint_to_PIL(uint_img):
    pass

def PIL_to_float(pil_img):
    pass

def uint_to_float(pil_img):
    pass

def tile_images(imgs, size=(6, 6)):
    imgs = imgs[:size[0]*size[1], :, :, :]
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[1]+i, :, :, :]
    return all_images

def visualize_samples(images, name="results/test.png", layout=[5,5], vrange=[-1., 1.]):
    images = (images - vrange[0]) / (vrange[1]-vrange[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = tile_images(images, size=layout)
    if name is None:
        return view
    view = Image.fromarray(view, 'RGB')
    view.save(name)

def broadcast_masks_tf(masks, num_channels=None, batch_size=None):
    if num_channels is not None:
        masks = tf.stack([masks for i in range(num_channels)], axis=-1)
    if batch_size is not None:
        masks = tf.stack([masks for i in range(batch_size)], axis=0)
    return masks

def broadcast_masks_np(masks, num_channels=None, batch_size=None):
    if num_channels is not None:
        masks = np.stack([masks for i in range(num_channels)], axis=-1)
    if batch_size is not None:
        masks = np.stack([masks for i in range(batch_size)], axis=0)
    return masks


def get_trainable_variables(flist, filter_type="in"):
    all_vs = tf.trainable_variables()
    if filter_type=="in":
        vs = []
        for s in flist:
            vs += [p for p in all_vs if s in p.name]
    elif filter_type=="not in":
        vs = all_vs
        for s in flist:
            vs = [p for p in vs if s not in p.name]
    return vs

def get_nonlinearity(name):
    if name=="relu":
        return tf.nn.relu
    elif name=="elu":
        return tf.nn.elu
    elif name=='tanh':
        return tf.nn.tanh
    elif name=='sigmoid':
        return tf.sigmoid


class Recorder(object):

    def __init__(self, dict={}, config_str="config not given", log_file="temp"):
        self.dict = dict
        self.keys = self.dict.keys()
        self.fetches = self.__fetches(self.keys)
        self.cur_values = []
        self.epoch_values = []
        self.past_epoch_stats = []
        self.num_epoches = 0
        self.log_file = log_file
        with open(self.log_file, "w") as f:
            f.write(config_str+"\n")

    def __fetches(self, keys):
        fetches = []
        for key in keys:
            fetches.append(self.dict[key])
        return fetches

    def evaluate(self, sess, feed_dict):
        self.cur_values = sess.run(self.fetches, feed_dict=feed_dict)
        self.epoch_values.append(self.cur_values)

    def finish_epoch_and_display(self, keys=None, time=0., log=True):
        epoch_values = np.array(self.epoch_values)
        stats = np.mean(epoch_values, axis=0)
        self.past_epoch_stats.append(stats)
        s = self.__display(stats, keys, time)
        print(s)
        sys.stdout.flush()
        with open(self.log_file, "a") as f:
            f.write(s+"\n")
        self.epoch_values = []
        self.num_epoches += 1

    def __display(self, stats, keys=None, time=0.):
        if keys is None:
            keys = self.keys
        results = {}
        for k, s in zip(self.keys, stats):
            results[k] = s
        ret_str = "* epoch {0} {1} -- ".format(self.num_epoches, "{"+"%0.2f"%time+"s}")
        for key in keys:
            ret_str += "{0}:{1:.3f}   ".format(key, results[key])
        return ret_str
