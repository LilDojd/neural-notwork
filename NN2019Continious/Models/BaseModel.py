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

import os
from functools import reduce
import numpy as np
import tensorflow as tf

from batch_factory import get_batch

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.90


class BaseModel:
    """Base model with training and testing procedures"""

    def __init__(self,
                 reg_fact,
                 learning_rate,
                 model_checkpoint_path,
                 max_to_keep, *args, **kwargs):

        self.model_checkpoint_path = model_checkpoint_path

        # Define the model though subclass
        self._init_model(*args, **kwargs)

        # Set weighted loss function (weight is hardcoded)
        logits = self.layers[-1]['dense']
        self.entropy = tf.squared_difference(logits, self.y)
        self.regularization = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith("b")]) * reg_fact

        self.loss = tf.reduce_mean(self.entropy) + self.regularization

        print("Number of parameters: ",
              sum(reduce(lambda x, y: x * y, v.get_shape().as_list()) for v in tf.trainable_variables()))

        # Set the optimizer #
        self.train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(
            self.loss)

        # Session and saver
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.session = tf.Session()

        self.mean = 0
        self.std = 1

        # Initialize variables
        tf.global_variables_initializer().run(session=self.session)
        print("Variables initialized")

        writer = tf.summary.FileWriter("/home/domain/yawner/2019/log/2")
        writer.add_graph(self.session.graph)

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def print_layer(layers, idx, name):
        """"Method for print architecture during model construction"""

        if layers[-1][name].get_shape()[0].value is None:
            size = int(np.prod(layers[-1][name].get_shape()[1:]))
        else:
            size = int(np.prod(layers[-1][name].get_shape()))

        print("layer %2d (high res) - %10s: %s [size %s]" % (
            len(layers), name, layers[idx][name].get_shape(), "{:,}".format(size)))

    @staticmethod
    def create_dense_layer(index,
                           input,
                           output_size):
        """Method for creating a dense layer"""

        reshaped_input = tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

        if output_size == -1:
            output_size = reshaped_input.get_shape().as_list()[1]

        W = tf.Variable(tf.truncated_normal([reshaped_input.get_shape().as_list()[1], output_size], stddev=0.1),
                        name="W%d" % index)
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name="b%d" % index)
        dense = tf.nn.bias_add(tf.matmul(reshaped_input, W), b)

        return {'W': W, 'b': b, 'dense': dense}

    def train(self,
              train_batch_factory,
              num_passes=100,
              max_batch_size=1000,
              subbatch_max_size=25,
              validation_batch_factory=None,
              output_interval=10,
              dropout_keep_prob=0.5,
              stdict=None):

        if stdict is None:
            stdict = {'mean': 0, 'std': 1}

        self.mean = stdict['mean']
        self.std = stdict['std']

        print("dropout keep probability: ", dropout_keep_prob)

        # Training loop
        iteration = 0

        with self.session.as_default():

            for i in range(num_passes):
                more_data = True

                while more_data:

                    batch, gradient_batch_sizes = train_batch_factory.next(max_batch_size,
                                                                           subbatch_max_size=subbatch_max_size,
                                                                           enforce_protein_boundaries=False)
                    more_data = (train_batch_factory.feature_index != 0)

                    grid_matrix = batch["high_res"]

                    vals = (batch["model_output"] - self.mean) / self.std

                    for sub_iteration, (index, length) in enumerate(
                            zip(np.cumsum(gradient_batch_sizes) - gradient_batch_sizes, gradient_batch_sizes)):
                        grid_matrix_batch, vals_batch = get_batch(index, index + length, grid_matrix, vals)

                        feed_dict = dict({self.x_high_res: grid_matrix_batch,
                                          self.y: vals_batch,
                                          self.dropout_keep_prob: dropout_keep_prob})

                        _, loss_value = self.session.run([self.train_step, self.loss], feed_dict=feed_dict)

                        print("[%d, %d, %02d] loss = %f" % (i, iteration, sub_iteration, loss_value))

                    if (iteration + 1) % output_interval == 0:

                        print("[%d, %d] Report%s (training batch):" % (
                            i, iteration, self.output_size))

                        R_training_batch, adjR, loss_training_batch = self.metrics(batch, gradient_batch_sizes)
                        print("SCORE TRAINING:", f"R-score: {R_training_batch} Adjusted R-score: {adjR}")
                        print("[%d, %d] loss (training batch) = %f" % (i, iteration, loss_training_batch))

                        validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(
                            validation_batch_factory.data_size(),
                            subbatch_max_size=subbatch_max_size,
                            enforce_protein_boundaries=False)
                        print(
                            "[%d, %d] Report%s (validation set):" % (i, iteration, self.output_size))
                        R_validation_batch, adjRval, loss_validation = self.metrics(validation_batch,
                                                                                    validation_gradient_batch_sizes)
                        print("SCORE TRAINING:", f"R-score: {R_validation_batch} Adjusted R-score: {adjRval}")
                        print("[%d, %d] loss (validation set) = %f" % (i, iteration, loss_validation))

                        self.save(self.model_checkpoint_path, iteration)

                    iteration += 1

    def save(self, checkpoint_path, step):

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.saver.save(self.session, os.path.join(checkpoint_path, 'model.ckpt'), global_step=step)
        print("Model saved")

    def restore(self, checkpoint_path, step=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            if step is None or step == -1:
                print("Restoring from: last checkpoint")
                self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
            else:
                checkpoint_file = checkpoint_path + ("/model.ckpt-%d" % step)
                print("Restoring from:", checkpoint_file)
                self.saver.restore(self.session, checkpoint_file)
        else:
            print("Could not load file")

    def _infer(self, batch, gradient_batch_sizes, var, include_output=False):
        global labels
        grid_matrix = batch["high_res"]

        if include_output:
            labels = (batch["model_output"] - self.mean) / self.std

        results = []

        for index, length in zip(np.cumsum(gradient_batch_sizes) - gradient_batch_sizes, gradient_batch_sizes):
            grid_matrix_batch, = get_batch(index, index + length, grid_matrix)

            feed_dict = {self.x_high_res: grid_matrix_batch, self.dropout_keep_prob: 1.0}

            if include_output:
                vals_batch, = get_batch(index, index + length, labels)
                feed_dict[self.y] = vals_batch

            results.append(self.session.run(var, feed_dict=feed_dict))

        return results

    def infer(self, batch, gradient_batch_sizes):
        results = self._infer(batch, gradient_batch_sizes, var=self.layers[-1]['activation'], include_output=False)
        return np.concatenate(results)

    def metrics(self, batch, gradient_batch_sizes, return_raw=False):

        y = batch["model_output"]
        results = self._infer(batch, gradient_batch_sizes, var=[self.layers[-1]['dense'], self.entropy],
                              include_output=True)
        y_, entropies = list(map(np.concatenate, list(zip(*results))))
        print(entropies, "ENTR")
        labes = self.std * y.T + self.mean
        print(labes)
        pred = self.std * y_.T + self.mean
        print(pred)
        SS_Residual = np.sum((labes - pred) ** 2)
        SS_Total = np.sum((labes - np.mean(labes)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        adj_r_squared = 1 - (1 - r_squared) * (len(labes) - 1) / (len(labes) - y.shape[1] - 1)

        regularization = self.session.run(self.regularization, feed_dict={})
        loss = np.mean(entropies) + regularization

        if return_raw:
            return loss, entropies, regularization
        else:
            return r_squared, adj_r_squared, loss

