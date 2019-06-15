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
from sklearn.metrics import classification_report as report

from batch_factory import get_batch

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95


logpath = "/home/domain/yawner/2019/log/train/spherical/1406bitsyVALID/"


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{name}', mean)
        with tf.name_scope(f'stddev_{name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{name}', stddev)
        tf.summary.scalar(f'max_{name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{name}', tf.reduce_min(var))
        tf.summary.histogram(f'histogram_{name}', var)


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
        with tf.name_scope('loss_and_entropy'):
            with tf.name_scope('entropy'):
                self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y, name="Crossentropy")
            with tf.name_scope('regularisation'):
                self.regularization = tf.add_n(
                         [tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith("b")]) * reg_fact
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(self.entropy) + self.regularization

        tf.summary.histogram('entropy', self.entropy)
        tf.summary.scalar('loss', self.loss)

        print("Number of parameters: ",
              sum(reduce(lambda x, y: x * y, v.get_shape().as_list()) for v in tf.trainable_variables()))

        # Set the optimizer #
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(
                self.loss)

        # Session and saver
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.session = tf.Session()

        # self.preds = self.session.run(self.x_high_res)
        #
        # self.correct_pred = tf.equal(tf.argmax(self.peds, 1), tf.argmax(self.y, 1))
        #
        # with tf.name_scope('accuracy'):
        #     self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # tf.summary.scalar('accuracy', self.accuracy)
        self.accuracy = None
        self.accuracy_summary = tf.Summary()
        self.accuracy_summary.value.add(tag='accuracy', simple_value=self.accuracy)

        self.validation_accuracy = None
        self.validation_accuracy_summary = tf.Summary()
        self.validation_accuracy_summary.value.add(tag='accuracy', simple_value=self.accuracy)

        # Initialize variables
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(f"{logpath}/plot_train")
        self.validation_writer = tf.summary.FileWriter(f"{logpath}/plot_val")
        self.writer.add_graph(self.session.graph)
        tf.global_variables_initializer().run(session=self.session)
        print("Variables initialized")

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
        with tf.name_scope('dense_%d' % index):
            with tf.name_scope('reshape_%d' % index):
                reshaped_input = tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

            if output_size == -1:
                output_size = reshaped_input.get_shape().as_list()[1]
            with tf.name_scope('weights_dense_%d' % index):
                W = tf.Variable(tf.truncated_normal([reshaped_input.get_shape().as_list()[1], output_size], stddev=0.1),
                                name="W%d" % index)
            variable_summaries(W, f"W{index}")
            with tf.name_scope('bias_dense_%d' % index):
                b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name="b%d" % index)
            variable_summaries(b, f"b{index}")
            dense = tf.nn.bias_add(tf.matmul(reshaped_input, W), b)

        return {'W': W, 'b': b, 'dense': dense}

    def train(self,
              train_batch_factory,
              num_passes=100,
              max_batch_size=1000,
              subbatch_max_size=25,
              validation_batch_factory=None,
              output_interval=10,
              dropout_keep_prob=0.5):

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

                    vals = batch["model_output"]

                    for sub_iteration, (index, length) in enumerate(
                            zip(np.cumsum(gradient_batch_sizes) - gradient_batch_sizes, gradient_batch_sizes)):
                        grid_matrix_batch, vals_batch = get_batch(index, index + length, grid_matrix, vals)

                        feed_dict = dict({self.x_high_res: grid_matrix_batch,
                                          self.y: vals_batch,
                                          self.dropout_keep_prob: dropout_keep_prob})

                        summary, _, loss_value = self.session.run([self.merged, self.train_step, self.loss], feed_dict=feed_dict)

                        print("[%d, %d, %02d] loss = %f" % (i, iteration, sub_iteration, loss_value))

                    self.writer.add_summary(summary, iteration)

                    accuracy = self.get_accuracy(vals_batch, grid_matrix_batch)
                    self.accuracy_summary.value[0].simple_value = accuracy
                    self.writer.add_summary(self.accuracy_summary, iteration)
                    print("Current training accuracy:", accuracy)

                    # valid_batch, _ = validation_batch_factory.next(
                    #     validation_batch_factory.data_size(),
                    #     subbatch_max_size=subbatch_max_size,
                    #     enforce_protein_boundaries=False)
                    # valid_vals_batch = valid_batch['model_output']
                    # valid_grid_matr_batch = valid_batch['high_res']
                    # val_accuracy = self.get_accuracy(valid_vals_batch, valid_grid_matr_batch)
                    # print("Current validation accuracy:", val_accuracy)
                    # self.validation_accuracy_summary.value[0].simple_value = val_accuracy
                    # self.validation_writer.add_summary(self.validation_accuracy_summary, iteration)

                    if (iteration + 1) % output_interval == 0:

                        print("[%d, %d] Report%s (training batch):" % (
                            i, iteration, self.output_size))
                        Q_training_batch, loss_training_batch = self.F_score_and_loss(batch, gradient_batch_sizes)
                        print("SCORE TRAINING:", Q_training_batch)
                        print("[%d, %d] loss (training batch) = %f" % (i, iteration, loss_training_batch))

                        validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(
                            validation_batch_factory.data_size(),
                            subbatch_max_size=subbatch_max_size,
                            enforce_protein_boundaries=False)
                        print(
                            "[%d, %d] Report%s (validation set):" % (i, iteration, self.output_size))
                        Q_validation, loss_validation = self.F_score_and_loss(validation_batch,
                                                                              validation_gradient_batch_sizes)
                        self.validation_accuracy_summary.value[0].simple_value = Q_validation
                        self.validation_writer.add_summary(self.validation_accuracy_summary, iteration)

                        print("SCORE VALIDATION:", Q_validation)
                        print("[%d, %d] loss (validation set) = %f" % (i, iteration, loss_validation))

                        self.save(self.model_checkpoint_path, iteration)

                    self.writer.flush()
                    self.validation_writer.flush()

                    iteration += 1

            self.writer.close()
            self.validation_writer.close()

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
            labels = batch["model_output"]

        results = []

        for index, length in zip(np.cumsum(gradient_batch_sizes) - gradient_batch_sizes, gradient_batch_sizes):
            grid_matrix_batch, = get_batch(index, index + length, grid_matrix)

            feed_dict = {self.x_high_res: grid_matrix_batch, self.dropout_keep_prob: 1.0}

            if include_output:
                labels_batch, = get_batch(index, index + length, labels)
                feed_dict[self.y] = labels_batch

            results.append(self.session.run(var, feed_dict=feed_dict))

        return results

    def infer(self, batch, gradient_batch_sizes):
        results = self._infer(batch, gradient_batch_sizes, var=self.layers[-1]['activation'], include_output=False)
        return np.concatenate(results)

    def F_score_and_loss(self, batch, gradient_batch_sizes, return_raw=False, return_Q=True):
        y = batch["model_output"]
        y_argmax = np.argmax(y, 1)
        results = self._infer(batch, gradient_batch_sizes, var=[self.layers[-1]['dense'], self.entropy],
                              include_output=True)
        y_, entropies = list(map(np.concatenate, list(zip(*results))))
        predictions = np.argmax(y_, 1)

        identical = (predictions == y_argmax)
        Q_accuracy = np.mean(identical)
        print(np.vstack((y_argmax, predictions)))
        F_score = report(y_argmax, predictions, target_names=['Worse', 'Same', 'Better'])

        regularization = self.session.run(self.regularization, feed_dict={})
        loss = np.mean(entropies) + regularization

        if return_raw:
            return loss, identical, entropies, regularization
        elif return_Q:
            return Q_accuracy, loss
        elif not return_Q:
            return F_score, loss

    def get_accuracy(self, vals_batch, grid_mat_batch):
        labes = np.argmax(vals_batch, 1)
        preds = self.session.run(self.layers[-1]['dense'], {self.x_high_res: grid_mat_batch,
                                                            self.dropout_keep_prob: 1.0,
                                                            self.y: vals_batch})
        y_ = np.argmax(preds, 1)
        idents = (labes == y_)
        return np.mean(idents)