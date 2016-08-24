import time
import inspect
import os
import urllib.request

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]
VGG16_URL = 'http://maxwell.cs.umass.edu/hsu/vgg16.npy'
VGG19_URL = 'http://maxwell.cs.umass.edu/hsu/vgg19.npy'

USE_FP16 = False # Not yet fully supported


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
        self.loaded_vars = set({})
        self.regularized_vars = set({})
        self.model = model
        print("model file ({}) loaded. ".format(npy_path))

    def build(self, net_input, layer_range=(0, 9), name='', summary=False, weight_decay=0.0, use_variable=False,
              num_classes=None, keep_original_data=True):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0.0, 255.0]
        :param num_classes: if not None, a fc8 with num_classes randomly initialized nodes will be used instead
        :param use_variable: if True, variables (instead of constants) will be used for weights
        :param layer_range: only layers within this range will be added (pre-processing is layer#0 and prob is layer#9)
        """

        pfx = '' if name == '' else (name + '/')
        self.num_classes = num_classes
        net = net_input
        params = []

        start_time = time.time()
        print("building {} model ... ".format(self.model), end='')

        # Layer #0: Convert RGB to BGR
        if layer_range[0] <= 0 <= layer_range[1]:
            red, green, blue = tf.split(3, 3, net)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
            net = bgr

        # Layer #1: CONV1
        if layer_range[0] <= 1 <= layer_range[1]:
            self._conv1_1, self._conv1_1_params = self.conv_layer(net, name, 'conv1_1', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv1_2, self._conv1_2_params = self.conv_layer(self._conv1_1, name, 'conv1_2', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self.pool1 = self.max_pool(self._conv1_2, pfx + 'pool1')
            self.conv1, self.conv1_params = self._conv1_2, self._conv1_1_params + self._conv1_2_params
            net = self.pool1
            params += self.conv1_params

        # Layer #2: CONV2
        if layer_range[0] <= 2 <= layer_range[1]:
            self._conv2_1, self._conv2_1_params = self.conv_layer(net, name, 'conv2_1', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv2_2, self._conv2_2_params = self.conv_layer(self._conv2_1, name, 'conv2_2', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self.pool2 = self.max_pool(self._conv2_2, pfx + 'pool2')
            self.conv2, self.conv2_params = self._conv2_2, self._conv2_1_params + self._conv2_2_params
            net = self.pool2
            params += self.conv2_params

        # Layer #3: CONV3
        if layer_range[0] <= 3 <= layer_range[1]:
            self._conv3_1, self._conv3_1_params = self.conv_layer(net, name, 'conv3_1', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv3_2, self._conv3_2_params = self.conv_layer(self._conv3_1, name, 'conv3_2', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv3_3, self._conv3_3_params = self.conv_layer(self._conv3_2, name, 'conv3_3', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self.conv3_params = self._conv3_1_params + self._conv3_2_params + self._conv3_3_params
            if self.model == 'vgg19':
                self._conv3_4, self._conv3_4_params = self.conv_layer(self._conv3_3, name, 'conv3_4', summary=summary,
                                                                      wd=weight_decay, use_variable=use_variable)
                self.conv3 = self._conv3_4
                self.conv3_params += self._conv3_4_params
            else:
                self.conv3 = self._conv3_3
            self.pool3 = self.max_pool(self.conv3, pfx + 'pool3')
            net = self.pool3
            params += self.conv3_params

        # Layer #4: CONV4
        if layer_range[0] <= 4 <= layer_range[1]:
            self._conv4_1, self._conv4_1_params = self.conv_layer(net, name, 'conv4_1', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv4_2, self._conv4_2_params = self.conv_layer(self._conv4_1, name, 'conv4_2', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv4_3, self._conv4_3_params = self.conv_layer(self._conv4_2, name, 'conv4_3', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self.conv4_params = self._conv4_1_params + self._conv4_2_params + self._conv4_3_params
            if self.model == 'vgg19':
                self._conv4_4, self._conv4_4_params = self.conv_layer(self._conv4_3, name, 'conv4_4', summary=summary,
                                                                      wd=weight_decay, use_variable=use_variable)
                self.conv4 = self._conv4_4
                self.conv4_params += self._conv4_4_params
            else:
                self.conv4 = self._conv4_3
            self.pool4 = self.max_pool(self.conv4, pfx + 'pool4')
            net = self.pool4
            params += self.conv4_params

        # Layer #5: CONV5
        if layer_range[0] <= 5 <= layer_range[1]:
            self._conv5_1, self._conv5_1_params = self.conv_layer(net, name, 'conv5_1', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv5_2, self._conv5_2_params = self.conv_layer(self._conv5_1, name, 'conv5_2', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self._conv5_3, self._conv5_3_params = self.conv_layer(self._conv5_2, name, 'conv5_3', summary=summary,
                                                                  wd=weight_decay, use_variable=use_variable)
            self.conv5_params = self._conv5_1_params + self._conv5_2_params + self._conv5_3_params
            if self.model == 'vgg19':
                self._conv5_4, self._conv5_4_params = self.conv_layer(self._conv5_3, name, 'conv5_4', summary=summary,
                                                                      wd=weight_decay, use_variable=use_variable)
                self.conv5 = self._conv5_4
                self.conv5_params += self._conv5_4_params
            else:
                self.conv5 = self._conv5_3
            self.pool5 = self.max_pool(self.conv5, pfx + 'pool5')
            net = self.pool5
            params += self.conv5_params

        # Layer #6: FC6
        if layer_range[0] <= 6 <= layer_range[1]:
            self.fc6, self.fc6_params = self.fc_layer(net, name, 'fc6', summary=summary, wd=weight_decay,
                                                      use_variable=use_variable)
            net = self.fc6
            params += self.fc6_params

        # Layer #7: FC7
        if layer_range[0] <= 7 <= layer_range[1]:
            self.fc7, self.fc7_params = self.fc_layer(net, name, 'fc7', summary=summary, wd=weight_decay,
                                                      use_variable=use_variable)
            net = self.fc7
            params += self.fc7_params

        # Layer #8: FC8
        if layer_range[0] <= 8 <= layer_range[1]:
            self.fc8, self.fc8_params = self.fc_layer(net, name, 'fc8', relu=False, summary=summary, wd=weight_decay,
                                                      use_variable=use_variable, random_init=(num_classes is not None))
            net = self.fc8
            params += self.fc8_params

        # Layer #9: PROBABILITY
        if layer_range[0] <= 9 <= layer_range[1]:
            self.prob = tf.nn.softmax(net, name=pfx + 'prob')
            net = self.prob

        if not keep_original_data:
            self.data_dict = None
        print("done! (%d secs)" % (time.time() - start_time))

        return net, params

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, pfx, name, relu=True, summary=False, wd=0.0, use_variable=False, random_init=False):
        with tf.variable_scope(name):
            filt = self.get_weight(name, use_variable=use_variable, random_init=random_init)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, use_variable=use_variable, random_init=random_init)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                relu = tf.nn.relu(bias)

            if summary:
                full_name = (name if pfx == '' else (pfx + '/' + name))
                tf.histogram_summary('activations/{}'.format(full_name), relu)
                tf.scalar_summary('sparsity/{}'.format(full_name), tf.nn.zero_fraction(relu))

            if wd > 0:
                var_name = tf.get_variable_scope().name + '/weights'
                if var_name not in self.regularized_vars:
                    weight_decay = tf.mul(wd, tf.nn.l2_loss(filt), name='weight_loss')
                    tf.add_to_collection('regularizers', weight_decay)
                    self.regularized_vars.add(var_name)
            return relu, [filt, conv_biases]

    def fc_layer(self, bottom, pfx, name, relu=True, summary=False, wd=0.0, use_variable=False, random_init=False):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_weight(name, use_variable=use_variable, random_init=random_init)
            biases = self.get_bias(name, use_variable=use_variable, random_init=random_init)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            if relu:
                fc = tf.nn.relu(fc)

            if summary:
                full_name = (name if pfx == '' else (pfx + '/' + name))
                tf.histogram_summary('activations/{}'.format(full_name), fc)
                tf.scalar_summary('sparsity/{}'.format(full_name), tf.nn.zero_fraction(fc))

            if wd > 0:
                var_name = tf.get_variable_scope().name + '/weights'
                if var_name not in self.regularized_vars:
                    weight_decay = tf.mul(wd, tf.nn.l2_loss(weights), name='weight_loss')
                    tf.add_to_collection('regularizers', weight_decay)
                    self.regularized_vars.add(var_name)
            return fc, [weights, biases]

    def get_weight(self, name, use_variable=False, random_init=False):
        if use_variable or random_init:
            var_name = tf.get_variable_scope().name + '/weights'
            if var_name in self.loaded_vars:
                tf.get_variable_scope().reuse_variables()
            else:
                self.loaded_vars.add(var_name)
            if random_init:
                shape = list(self.data_dict[name][0].shape)
                if name == 'fc8' and self.num_classes is not None:
                    shape[-1] = self.num_classes
                return tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(
                    stddev=0.01, dtype=(tf.float16 if USE_FP16 else tf.float32)))
            else:
                return tf.get_variable('weights', initializer=tf.constant(self.data_dict[name][0]))
        else:
            return tf.constant(self.data_dict[name][0], name='weights')

    def get_bias(self, name, use_variable=False, random_init=False):
        if use_variable or random_init:
            var_name = tf.get_variable_scope().name + '/biases'
            if var_name in self.loaded_vars:
                tf.get_variable_scope().reuse_variables()
            else:
                self.loaded_vars.add(var_name)
            if random_init:
                shape = list(self.data_dict[name][1].shape)
                if name == 'fc8' and self.num_classes is not None:
                    shape[-1] = self.num_classes
                return tf.get_variable('biases', shape=shape, initializer=tf.constant_initializer(
                    dtype=(tf.float16 if USE_FP16 else tf.float32)))
            else:
                return tf.get_variable('biases', initializer=tf.constant(self.data_dict[name][1]))
        else:
            return tf.constant(self.data_dict[name][1], name='biases')
