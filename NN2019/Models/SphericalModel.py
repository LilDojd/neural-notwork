# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import tensorflow as tf

from .SphericalBaseModel import SphericalBaseModel


class SphericalModel(SphericalBaseModel):

    def _init_model(self,
                    r_size_high_res,
                    theta_size_high_res,
                    phi_size_high_res,
                    channels_high_res,
                    output_size):
        with tf.name_scope("inputs"):

            self.output_size = output_size
            self.x_high_res = tf.placeholder(tf.float32, [None, r_size_high_res, theta_size_high_res, phi_size_high_res,
                                             channels_high_res], name="X_placeholder")
            self.y = tf.placeholder(tf.float32, [None, output_size], name="Y_placeholder")
            self.dropout_keep_prob = tf.placeholder(tf.float32)

        ### LAYER 0 ###
        self.layers = [{'input': self.x_high_res}]
        self.print_layer(self.layers, -1, 'input')

        ### LAYER 1 ###
        self.layers.append({})
        with tf.name_scope("1st_conv"):
            self.layers[-1].update(self.create_spherical_conv_layer(len(self.layers) - 1,
                                                                    self.x_high_res,
                                                                    window_size_r=3,
                                                                    window_size_theta=5,
                                                                    window_size_phi=5,
                                                                    channels_out=16,
                                                                    stride_r=1,
                                                                    stride_theta=2,
                                                                    stride_phi=2,
                                                                    padding='VALID'))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'], name="relu1")
            self.layers[-1].update(self.create_spherical_avgpool_layer(len(self.layers) - 1,
                                                                       self.layers[-1]['activation'],
                                                                       ksize=[1, 1, 3, 3, 1],
                                                                       strides=[1, 1, 2, 2, 1]))

        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')

        ### LAYER 2 ###
        with tf.name_scope("2nd_conv"):
            self.layers.append({})
            self.layers[-1].update(self.create_spherical_conv_layer(len(self.layers) - 1,
                                                                    self.layers[-2]['pool'],
                                                                    window_size_r=3,
                                                                    window_size_theta=3,
                                                                    window_size_phi=3,
                                                                    channels_out=32,
                                                                    stride_r=1,
                                                                    stride_theta=1,
                                                                    stride_phi=1,
                                                                    padding='VALID'))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'], name="relu2")
            self.layers[-1].update(self.create_spherical_avgpool_layer(len(self.layers) - 1,
                                                                       self.layers[-1]['activation'],
                                                                       ksize=[1, 3, 3, 3, 1],
                                                                       strides=[1, 2, 1, 2, 1]))
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')

        ### LAYER 3 ###
        self.layers.append({})
        with tf.name_scope("3rd_conv"):
            self.layers[-1].update(self.create_spherical_conv_layer(len(self.layers) - 1,
                                                                    self.layers[-2]['pool'],
                                                                    window_size_r=3,
                                                                    window_size_theta=3,
                                                                    window_size_phi=3,
                                                                    channels_out=64,
                                                                    stride_r=1,
                                                                    stride_theta=1,
                                                                    stride_phi=1,
                                                                    padding='VALID'))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'], name="relu3")
            self.layers[-1].update(self.create_spherical_avgpool_layer(len(self.layers) - 1,
                                                                       self.layers[-1]['activation'],
                                                                       ksize=[1, 1, 3, 3, 1],
                                                                       strides=[1, 1, 1, 2, 1]))

        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')

        ### LAYER 4 ###
        with tf.name_scope("4th_conv"):
            self.layers.append({})
            self.layers[-1].update(self.create_spherical_conv_layer(len(self.layers) - 1,
                                                                    self.layers[-2]['pool'],
                                                                    window_size_r=3,
                                                                    window_size_theta=2,
                                                                    window_size_phi=3,
                                                                    channels_out=128,
                                                                    stride_r=1,
                                                                    stride_theta=1,
                                                                    stride_phi=1,
                                                                    padding='VALID'))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'], name="relu4")
            self.layers[-1].update(self.create_spherical_avgpool_layer(len(self.layers) - 1,
                                                                       self.layers[-1]['activation'],
                                                                       ksize=[1, 1, 1, 3, 1],
                                                                       strides=[1, 1, 2, 1, 1]))
        self.print_layer(self.layers, -1, 'activation')
        self.print_layer(self.layers, -1, 'pool')

        ### LAYER 5 ###
        self.layers.append({})
        with tf.name_scope("dense_layer1"):
            self.layers[-1].update(self.create_dense_layer(len(self.layers) - 1,
                                                           self.layers[-2]['pool'],
                                                           output_size=2048))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'], name="relu5")
            self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob, name="Dropout")
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 6 ###
        self.layers.append({})
        with tf.name_scope("dense_layer2"):
            self.layers[-1].update(self.create_dense_layer(len(self.layers) - 1,
                                                           self.layers[-2]['dropout'],
                                                           output_size=1024))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'], name="relu6")
            self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob, name="Dropout")
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'activation')

        ### LAYER 7 ###
        self.layers.append({})
        with tf.name_scope("Output_layer3"):
            self.layers[-1].update(self.create_dense_layer(len(self.layers) - 1,
                                                           self.layers[-2]['dropout'],
                                                           output_size=output_size))
            self.layers[-1]['activation'] = tf.nn.softmax(self.layers[-1]['dense'], name="Softmax_classification")
        self.print_layer(self.layers, -1, 'W')
        self.print_layer(self.layers, -1, 'dense')
