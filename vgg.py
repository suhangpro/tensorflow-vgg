import time
import inspect
import os
import urllib.request

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]
VGG16_URL = 'http://maxwell.cs.umass.edu/hsu/vgg16.npy'
VGG19_URL = 'http://maxwell.cs.umass.edu/hsu/vgg19.npy'


# noinspection PyAttributeOutsideInit
class VggVd:
    def __init__(self, model='vgg16', npy_path=None):
        if npy_path is None:
            path = inspect.getfile(VggVd)
            path = os.path.abspath(os.path.join(path, os.pardir))
            npy_path = os.path.join(path, "{}.npy".format(model))

        if not os.path.isfile(npy_path):
            if model == 'vgg16':
                npy_url = VGG16_URL
            else:
                npy_url = VGG19_URL
            start_time = time.time()
            print('downloading model file ({}) ... '.format(npy_url), end='', flush=True)
            urllib.request.urlretrieve(npy_url, npy_path)
            print('done! ({} secs)'.format(time.time() - start_time))
        self.data_dict = np.load(npy_path, encoding='latin1').item()
        self.model = model
        print("model file ({}) loaded. ".format(npy_path))

    def build(self, rgb, summary=False, weight_decay=0.0, train=False):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0.0, 255.0]
        """

        start_time = time.time()
        print("building {} model ... ".format(self.model), end='')

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # CONV1
        self._conv1_1, self._conv1_1_params = self.conv_layer(bgr, "conv1_1", summary=summary, wd=weight_decay, train=train)
        self._conv1_2, self._conv1_2_params = self.conv_layer(self._conv1_1, "conv1_2", summary=summary, wd=weight_decay, train=train)
        self.pool1 = self.max_pool(self._conv1_2, 'pool1')
        self.conv1, self.conv1_params = self._conv1_2, self._conv1_1_params + self._conv1_2_params

        # CONV2
        self._conv2_1, self._conv2_1_params = self.conv_layer(self.pool1, "conv2_1", summary=summary, wd=weight_decay, train=train)
        self._conv2_2, self._conv2_2_params = self.conv_layer(self._conv2_1, "conv2_2", summary=summary, wd=weight_decay, train=train)
        self.pool2 = self.max_pool(self._conv2_2, 'pool2')
        self.conv2, self.conv2_params = self._conv2_2, self._conv2_1_params + self._conv2_2_params

        # CONV3
        self._conv3_1, self._conv3_1_params = self.conv_layer(self.pool2, "conv3_1", summary=summary, wd=weight_decay, train=train)
        self._conv3_2, self._conv3_2_params = self.conv_layer(self._conv3_1, "conv3_2", summary=summary, wd=weight_decay, train=train)
        self._conv3_3, self._conv3_3_params = self.conv_layer(self._conv3_2, "conv3_3", summary=summary, wd=weight_decay, train=train)
        self.conv3_params = self._conv3_1_params + self._conv3_2_params + self._conv3_3_params
        if self.model == 'vgg19':
            self._conv3_4, self._conv3_4_params = self.conv_layer(self._conv3_3, "conv3_4", summary=summary, wd=weight_decay, train=train)
            self.conv3 = self._conv3_4
            self.conv3_params += self._conv3_4_params
        else:
            self.conv3 = self._conv3_3
        self.pool3 = self.max_pool(self.conv3, 'pool3')

        # CONV4
        self._conv4_1, self._conv4_1_params = self.conv_layer(self.pool3, "conv4_1", summary=summary, wd=weight_decay, train=train)
        self._conv4_2, self._conv4_2_params = self.conv_layer(self._conv4_1, "conv4_2", summary=summary, wd=weight_decay, train=train)
        self._conv4_3, self._conv4_3_params = self.conv_layer(self._conv4_2, "conv4_3", summary=summary, wd=weight_decay, train=train)
        self.conv4_params = self._conv4_1_params + self._conv4_2_params + self._conv4_3_params
        if self.model == 'vgg19':
            self._conv4_4, self._conv4_4_params = self.conv_layer(self._conv4_3, "conv4_4", summary=summary, wd=weight_decay, train=train)
            self.conv4 = self._conv4_4
            self.conv4_params += self._conv4_4_params
        else:
            self.conv4 = self._conv4_3
        self.pool4 = self.max_pool(self.conv4, 'pool4')

        # CONV5
        self._conv5_1, self._conv5_1_params = self.conv_layer(self.pool4, "conv5_1", summary=summary, wd=weight_decay, train=train)
        self._conv5_2, self._conv5_2_params = self.conv_layer(self._conv5_1, "conv5_2", summary=summary, wd=weight_decay, train=train)
        self._conv5_3, self._conv5_3_params = self.conv_layer(self._conv5_2, "conv5_3", summary=summary, wd=weight_decay, train=train)
        self.conv5_params = self._conv5_1_params + self._conv5_2_params + self._conv5_3_params
        if self.model == 'vgg19':
            self._conv5_4, self._conv5_4_params = self.conv_layer(self._conv5_3, "conv5_4", summary=summary, wd=weight_decay, train=train)
            self.conv5 = self._conv5_4
            self.conv5_params += self._conv5_4_params
        else:
            self.conv5 = self._conv5_3
        self.pool5 = self.max_pool(self.conv5, 'pool5')

        # FC6
        self.fc6, self.fc6_params = self.fc_layer(self.pool5, "fc6", summary=summary, wd=weight_decay, train=train)

        # FC7
        self.fc7, self.fc7_params = self.fc_layer(self.fc6, "fc7", summary=summary, wd=weight_decay, train=train)

        # FC8
        self.fc8, self.fc8_params = self.fc_layer(self.fc7, "fc8", relu=False, summary=summary, wd=weight_decay, train=train)

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("done! (%d secs)" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, relu=True, summary=False, wd=0.0, train=False):
        with tf.variable_scope(name):
            filt = self.get_weight(name, use_variable=train)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, use_variable=train)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                relu = tf.nn.relu(bias)

            if summary:
                tf.histogram_summary('activations', relu)
                tf.scalar_summary('sparsity', tf.nn.zero_fraction(relu))

            if wd > 0:
                weight_decay = wd * tf.nn.l2_loss(filt)
                tf.add_to_collection('losses', weight_decay)
            return relu, [filt, conv_biases]

    def fc_layer(self, bottom, name, relu=True, summary=False, wd=0.0, train=False):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_weight(name, use_variable=train)
            biases = self.get_bias(name, use_variable=train)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            if relu:
                fc = tf.nn.relu(fc)

            if summary:
                tf.histogram_summary('activations', fc)
                tf.scalar_summary('sparsity', tf.nn.zero_fraction(fc))

            if wd > 0:
                weight_decay = wd * tf.nn.l2_loss(weights)
                tf.add_to_collection('losses', weight_decay)
            return fc, [weights, biases]

    def get_weight(self, name, use_variable=False):
        if use_variable:
            return tf.get_variable('weights', initializer=tf.constant(self.data_dict[name][0]))
        else:
            return tf.constant(self.data_dict[name][0], name='weights')

    def get_bias(self, name, use_variable=False):
        if use_variable:
            return tf.get_variable('biases', initializer=tf.constant(self.data_dict[name][1]))
        else:
            return tf.constant(self.data_dict[name][1], name='biases')
