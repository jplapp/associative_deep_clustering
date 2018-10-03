"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Utility functions for Association-based semisupervised training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.optimize import linear_sum_assignment
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

NO_FC_COLLECTION = "NO_FC_COLLECTION"

def show_sample_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()


def show_sample_img_inline(imgs):
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(1, max(len(imgs), 2))
    plt.axis('off')
    for ind, img in enumerate(imgs):
        axarr[ind].axis('off')
        axarr[ind].imshow(img.reshape(28, 28), cmap='gray')
    plt.show()


def show_samples(imgs, image_shape, scale=128., figsize=(8, 16)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1], sharex=True, sharey=True, figsize=figsize)
    for i in range(imgs.shape[0]):
        row = axes[i]
        for image, ax in zip(imgs[i], row):

            ax.imshow(np.array(image.reshape(image_shape) * scale + scale, np.uint8))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)


def create_input(input_images, input_labels, batch_size, shuffle=False):
    """Create preloaded data batch inputs.

    Args:
      input_images: 4D numpy array of input images.
      input_labels: 2D numpy array of labels.
      batch_size: Size of batches that will be produced.

    Returns:
      A list containing the images and labels batches.
    """
    if batch_size == -1:
        batch_size = input_labels.shape[0]
    if input_labels is not None:
        image, label = tf.train.slice_input_producer(
                [input_images, input_labels])
        if shuffle:
            return tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                          capacity=500, min_after_dequeue=100)
        else:
            return tf.train.batch([image, label], batch_size=batch_size)
    else:
        image = tf.train.slice_input_producer([input_images])
        return tf.train.batch(image, batch_size=batch_size)


def create_per_class_inputs(image_by_class, n_per_class, class_labels=None):
    """Create batch inputs with specified number of samples per class.

    Args:
      image_by_class: List of image arrays, where image_by_class[i] containts
          images sampled from the class class_labels[i].
      n_per_class: Number of samples per class in the output batch.
      class_labels: List of class labels. Equals to range(len(
      image_by_class)) if
          not provided.

    Returns:
      images: Tensor of n_per_class*len(image_by_class) images.
      labels: Tensor of same number of labels.
    """
    if class_labels is None:
        class_labels = np.arange(len(image_by_class))
    batch_images, batch_labels = [], []
    for images, label in zip(image_by_class, class_labels):
        labels = tf.fill([len(images)], label)
        images, labels = create_input(images, labels, n_per_class)
        batch_images.append(images)
        batch_labels.append(labels)
    return tf.concat(batch_images, 0), tf.concat(batch_labels, 0)


def create_per_class_inputs_sub_batch(image_by_class, n_per_class,
                                      class_labels=None):
    """Create batch inputs with specified number of samples per class.

    Args:
      image_by_class: List of image arrays, where image_by_class[i] containts
          images sampled from the class class_labels[i].
      n_per_class: Number of samples per class in the output batch.
      class_labels: List of class labels. Equals to range(len(
      image_by_class)) if
          not provided.

    Returns:
      images: Tensor of n_per_class*len(image_by_class) images.
      labels: Tensor of same number of labels.
    """
    batch_images, batch_labels = [], []
    if class_labels is None:
        class_labels = np.arange(len(image_by_class))

    for images, label in zip(image_by_class, class_labels):
        labels = np.ones(len(images)) * label

        indices = np.random.choice(range(0, len(images)), n_per_class)
        # todo make sure we don't miss a sample here

        batch_images.extend(images[indices])
        batch_labels.extend(labels[indices])

    batch_images = np.asarray(batch_images)
    batch_labels = np.asarray(batch_labels, np.int)

    imgs, lbls = create_input(batch_images, batch_labels, batch_size=20,
                              shuffle=True)

    return imgs, lbls


def sample_by_label(images, labels, n_per_label, num_labels, seed=None):
    """Extract equal number of sampels per class."""
    res = []
    rng = np.random.RandomState(seed=seed)
    for i in range(num_labels):
        a = images[labels == i]
        if n_per_label == -1:  # use all available labeled data
            res.append(a)
        else:  # use randomly chosen subset
            inds = rng.choice(len(a), n_per_label, False)
            res.append(a[inds])
    return res


def create_virt_emb(n, size):
    """Create virtual embeddings."""
    emb = slim.variables.model_variable(name='virt_emb',
                                        shape=[n, size],
                                        dtype=tf.float32,
                                        trainable=True,
                                        initializer=tf.random_normal_initializer(
                                                stddev=0.01))
    return emb


def confusion_matrix(labels, predictions, num_labels):
    """Compute the confusion matrix."""
    rows = []
    for i in range(num_labels):
        row = np.bincount(predictions[labels == i], minlength=num_labels)
        rows.append(row)
    return np.vstack(rows)


def softmax(x):
    maxes = np.amax(x, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(x - maxes)
    dist = e / np.sum(e, axis=1).reshape(maxes.shape[0], 1)
    return dist


def one_hot(a, depth):
    b = np.zeros((a.size, depth))
    b[np.arange(a.size), a] = 1
    return b


def logistic_growth(current_step, target, steps):
    assert target >= 0., 'Target value must be positive.'
    alpha = 5. / steps
    return target * (np.tanh(alpha * (current_step - steps / 2.)) + 1.) / 2.


def apply_envelope(type, step, final_weight, growing_steps, delay):
    assert growing_steps > 0, "Growing steps for envelope must be > 0."
    step = step - delay
    if step <= 0:
        return 0

    final_step = growing_steps + delay

    if type is None:
        value = final_weight

    elif type in ['sigmoid', 'sigmoidal', 'logistic', 'log']:
        value = logistic_growth(step, final_weight, final_step)

    elif type in ['linear', 'lin']:
        m = float(final_weight) / (
            growing_steps) if not growing_steps == 0.0 else 999.
        value = m * step
    else:
        raise NameError('Invalid type: ' + str(type))

    return np.clip(value, 0., final_weight)


from tensorflow.python.framework import ops

LOSSES_COLLECTION = ops.GraphKeys.LOSSES


def l1_loss(tensor, weight, scope=None):
    """Define a L1Loss, useful for regularize, i.e. lasso.
    Args:
      tensor: tensor to regularize.
      weight: tensor: scale the loss by this factor.
      scope: Optional scope for name_scope.
    Returns:
      the L1 loss op.
    """
    with tf.name_scope(scope, 'L1Loss', [tensor]):
        # weight = tf.convert_to_tensor(weight,
        #                              dtype=tensor.dtype.base_dtype,
        #                              name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
        tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the
    same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape *
    repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor


class SemisupModel(object):
    """Helper class for setting up semi-supervised training."""

    def __init__(self, model_func, num_labels, input_shape, test_in=None,
                 optimizer='adam', beta1=0.9, beta2=0.999, num_blocks=None,
                 emb_size=128, dropout_keep_prob=1, augmentation_function=None,
                 normalize_embeddings=False, resize_shape=None):
        """Initialize SemisupModel class.

        Creates an evaluation graph for the provided model_func.

        Args:
          model_func: Model function. It should receive a tensor of images as
              the first argument, along with the 'is_training' flag.
          num_labels: Number of taget classes.
          input_shape: List, containing input images shape in form
              [height, width, channel_num].
          test_in: None or a tensor holding test images. If None,
          a placeholder will
            be created.
        """

        self.num_labels = num_labels
        self.step = tf.train.get_or_create_global_step()
        self.ema = tf.train.ExponentialMovingAverage(0.99, self.step)
        self.emb_size = emb_size

        self.test_batch_size = 100

        self.model_func = model_func
        self.augmentation_function = augmentation_function
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_blocks = num_blocks
        self.dropout_keep_prob = dropout_keep_prob

        #test_in = None
        if test_in is not None:
            self.test_in = test_in
        elif resize_shape is not None:
            self.test_in = tf.placeholder(np.float32, [None,None,None,input_shape[2]],
                                          'test_in')
            test_in = tf.image.resize_images(self.test_in, input_shape[0:2])
        else:
            self.test_in = tf.placeholder(np.float32, [None] + input_shape,
                                          'test_in')
            test_in = self.test_in

        self.test_emb = self.image_to_embedding(test_in, is_training=False)
        if normalize_embeddings:
            self.test_emb = tf.nn.l2_normalize(self.test_emb, dim=1)
        self.test_logit = self.embedding_to_logit(self.test_emb,
                                                  is_training=False)

    def reset_optimizer(self, sess):
        optimizer_slots = [
            self.trainer.get_slot(var, name)
            for name in self.trainer.get_slot_names()
            for var in tf.model_variables()
            ]
        if isinstance(self.trainer, tf.train.AdamOptimizer):
            optimizer_slots.extend([
                self.trainer._beta1_power, self.trainer._beta2_power
                ])
        init_op = tf.variables_initializer(optimizer_slots)
        sess.run(init_op)

    def image_to_embedding(self, images, is_training=True):
        """Create a graph, transforming images into embedding vectors."""
        with tf.variable_scope('net', reuse=is_training):
            model = self.model_func(images, is_training=is_training,
                                    emb_size=self.emb_size,
                                    dropout_keep_prob=self.dropout_keep_prob,
                                    num_blocks=self.num_blocks,
                                    augmentation_function=self.augmentation_function)
            return model

    def embedding_to_logit(self, embedding, is_training=True):
        """Create a graph, transforming embedding vectors to logit class
        scores."""
        with tf.variable_scope('net', reuse=is_training):
            return slim.fully_connected(
                    embedding,
                    self.num_labels,
                    activation_fn=None,
                    weights_regularizer=slim.l2_regularizer(1e-4),
                    scope='logit_fc')

    def add_semisup_loss(self, a, b, labels, walker_weight=1.0,
                         visit_weight=1.0, match_scale=1.0, est_err=True, name='', use_proximity_loss=False):
        """Add semi-supervised classification loss to the model.

        The loss consists of two terms: "walker" and "visit".

        Args:
          a: [N, emb_size] tensor with supervised embedding vectors.
          b: [M, emb_size] tensor with unsupervised embedding vectors.
          labels : [N] tensor with labels for supervised embeddings.
          walker_weight: Weight coefficient of the "walker" loss.
          visit_weight: Weight coefficient of the "visit" loss.
        """

        equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
        equality_matrix = tf.cast(equality_matrix, tf.float32)
        p_target = (equality_matrix / tf.reduce_sum(
                equality_matrix, [1],
                keep_dims=True))

        match_ab = tf.matmul(a, b, transpose_b=True,
                             name='match_ab'+name) * match_scale
        p_ab = tf.nn.softmax(match_ab, name='p_ab'+name)
        p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba'+name)
        p_aba = tf.matmul(p_ab, p_ba, name='p_aba'+name)

        if est_err:
            self.create_walk_statistics(p_aba, equality_matrix)
            self.p_ab = p_ab
            self.p_ba = p_ba
            self.p_aba = p_aba

        loss_aba = tf.losses.softmax_cross_entropy(
                p_target,
                tf.log(1e-8 + p_aba),
                weights=walker_weight,
                scope='loss_aba'+name)
        self.loss_aba = loss_aba

        if use_proximity_loss:
            self.add_proximity_loss(p_ab, p_ba, visit_weight)
        else:
            self.add_visit_loss(p_ab, visit_weight)

        tf.summary.scalar('Loss_aba'+name, loss_aba)

        return loss_aba

    def add_logit_entropy(self, logits, weight=1.0, name=''):
        """
          (2): Logit entropy loss

          more clipping might be necessary

        """
        eps = 1e-8
        logits = tf.clip_by_value(logits, eps, 1 - eps)
        entropy = tf.reduce_mean(
                -tf.reduce_sum(logits * tf.log(logits), reduction_indices=[1]))
        entropy *= weight
        tf.add_to_collection(LOSSES_COLLECTION, entropy)

        tf.summary.scalar('logit_entropy' + name, entropy)

        return entropy

    def add_cluster_hardening_loss(self, logits, weight=1.0, name=''):
        """
          (1): Cluster hardening loss:
          Minimize KL distance between logits and logits^2

        """
        # kl_div = tf.contrib.distributions.kl(logits**2, logits)
        eps = 1e-8
        softmaxed_logits = tf.nn.softmax(logits)
        softmaxed_logits = tf.clip_by_value(softmaxed_logits, eps, 1 - eps)

        logits_squared = softmaxed_logits ** 2

        # logits_squared / logits = logits
        kl_div = tf.reduce_mean(
                tf.abs(tf.reduce_sum(logits_squared * tf.log(softmaxed_logits), 1)))

        kl_div *= weight

        tf.add_to_collection(LOSSES_COLLECTION, kl_div)

        tf.summary.scalar('cluster_hardening_kl_dist' + name, kl_div)

        return kl_div

    def add_semisup_loss_with_logits(self, a, b, logits, walker_weight=1.0,
                                     visit_weight=1.0, stop_gradient=False):
        """Add semi-supervised classification loss to the model.

        The loss consists of two terms: "walker" and "visit".

        Args:
          a: [N, emb_size] tensor with supervised embedding vectors.
          b: [M, emb_size] tensor with unsupervised embedding vectors.
          logits : [N, num_labels] tensor with logits of embedding probabilities
          walker_weight: Weight coefficient of the "walker" loss.
          visit_weight: Weight coefficient of the "visit" loss.
        """

        p = tf.nn.softmax(logits)

        equality_matrix = tf.matmul(p, p, transpose_b=True)
        equality_matrix = tf.cast(equality_matrix, tf.float32)
        p_target = (equality_matrix / tf.reduce_sum(
                equality_matrix, [1], keep_dims=True))

        match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
        p_ab = tf.nn.softmax(match_ab, name='p_ab')
        p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
        p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

        if stop_gradient:
            p_aba = tf.stop_gradient(p_aba)
            p_ab = tf.stop_gradient(p_ab)

        # self.create_walk_statistics(p_aba, equality_matrix)

        loss_aba = tf.losses.softmax_cross_entropy(
                p_target,
                tf.log(1e-8 + p_aba),
                weights=walker_weight,
                scope='loss_aba' + str(stop_gradient))

        self.add_visit_loss(p_ab, visit_weight, 'b' + str(stop_gradient))

        tf.summary.scalar('Loss_aba' + str(stop_gradient), loss_aba)

    def add_visit_loss(self, p, weight=1.0, name=''):
        """Add the "visit" loss to the model.

        Args:
          p: [N, M] tensor. Each row must be a valid probability distribution
              (i.e. sum to 1.0)
          weight: Loss weight.
        """
        visit_probability = tf.reduce_mean(
                p, [0], keep_dims=True, name='visit_prob' + name)
        t_nb = tf.shape(p)[1]
        visit_loss = tf.losses.softmax_cross_entropy(
                tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
                tf.log(1e-8 + visit_probability),
                weights=weight,
                scope='loss_visit' + name)

        tf.summary.scalar('Loss_Visit' + name, visit_loss)

    def add_proximity_loss(self, p_ba, p_ab, weight=1.0, name=''):
        """Add the "proximity" loss to the model."""

        print('adding proximity loss')
        p_bab = tf.matmul(p_ba, p_ab, name='p_bab')

        visit_probability = tf.reduce_mean(p_bab, [0], name='visit_prob_bab'+name, keep_dims=True)

        t_nb = tf.shape(p_ab)[1]
        visit_loss = tf.losses.softmax_cross_entropy(
                tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
                tf.log(1e-8 + visit_probability),
                weights=weight,
                scope='loss_visit' + name)

        tf.summary.scalar('Loss_Visit' + name, visit_loss)


    def add_logit_loss(self, logits, labels, weight=1.0, smoothing=0.0):
        """Add supervised classification loss to the model."""

        logit_loss = tf.losses.softmax_cross_entropy(
                tf.one_hot(labels, logits.get_shape()[-1]),
                logits,
                scope='loss_logit',
                weights=weight,
                label_smoothing=smoothing)
        return logit_loss

    def create_walk_statistics(self, p_aba, equality_matrix):
        """Adds "walker" loss statistics to the graph.

        Args:
          p_aba: [N, N] matrix, where element [i, j] corresponds to the
              probalility of the round-trip between supervised samples i and j.
              Sum of each row of 'p_aba' must be equal to one.
          equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
              i and j belong to the same class.
        """
        # Using the square root of the correct round trip probalilty as an
        # estimate
        # of the current classifier accuracy.
        # should be called estimated accuracy
        per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba),
                                               1) ** 0.5
        estimate_error = tf.reduce_mean(
                1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
        self.add_average(estimate_error)
        self.add_average(p_aba)

        self.estimate_error = estimate_error

        tf.summary.scalar('Stats_EstError', estimate_error)

    def add_average(self, variable):
        """Add moving average variable to the model."""
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             self.ema.apply([variable]))
        average_variable = tf.identity(
                self.ema.average(variable), name=variable.name[:-2] + '_avg')
        return average_variable

    def add_emb_regularization(self, embs, weight):
        """weight should be a tensor"""
        l1_loss(embs, weight)

    def add_emb_normalization(self, embs, weight, target=1):
        """weight should be a tensor"""
        l2n = tf.norm(embs, axis=1)
        self.l2n = l2n
        self.normalization = l1_loss((l2n - target) ** 2, weight)

    def create_train_op(self, learning_rate, gradient_multipliers=None, fc_stop_multiplier=None):
        """Create and return training operation."""

        slim.model_analyzer.analyze_vars(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                print_info=True)

        self.train_loss = tf.losses.get_total_loss()
        self.train_loss_average = self.add_average(self.train_loss)

        tf.summary.scalar('Learning_Rate', learning_rate)
        tf.summary.scalar('Loss_Total_Avg', self.train_loss_average)
        tf.summary.scalar('Loss_Total', self.train_loss)

        if self.optimizer == 'sgd':
            self.trainer = tf.train.MomentumOptimizer(
                    learning_rate, 0.9, use_nesterov=False)
        elif self.optimizer == 'adam':
            self.trainer = tf.train.AdamOptimizer(learning_rate,
                                                  beta1=self.beta1,
                                                  beta2=self.beta2)
        elif self.optimizer == 'rms':
            self.trainer = tf.train.RMSPropOptimizer(learning_rate,
                                                     momentum=0.9)
        else:
            print('unrecognized optimizer')

        self.train_op = slim.learning.create_train_op(self.train_loss,
                                                      self.trainer,
                                                      summarize_gradients=False,
                                                      gradient_multipliers=gradient_multipliers)

        # loss that should not influence last fc layer
        if len(tf.losses.get_losses(loss_collection=NO_FC_COLLECTION)):
          # todo maybe use second optimizer here
          #self.trainer2 = tf.train.AdamOptimizer(learning_rate,
          #                                     beta1=self.beta1,
          #                                     beta2=self.beta2)
          no_fc_loss = tf.reduce_sum(tf.losses.get_losses(loss_collection=NO_FC_COLLECTION))

          def is_not_logit(var):
            return 'logit_fc' not in var.name

          vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
          vars_no_logit = list(filter(is_not_logit, vars))

          self.train_op_sat = slim.learning.create_train_op(no_fc_loss,
                                                        self.trainer,
                                                        variables_to_train=vars_no_logit,
                                                        summarize_gradients=False)
        else:
          self.train_op_sat = tf.constant(0)

        return [self.train_op, self.train_op_sat]

    def calc_embedding(self, images, endpoint, sess, extra_feed_dict={}):
        """Evaluate 'endpoint' tensor for all 'images' using batches."""
        batch_size = self.test_batch_size
        emb = []
        for i in range(0, len(images), batch_size):
            feed_dict = {self.test_in: images[i: i + batch_size]}
            emb.append(endpoint.eval({**extra_feed_dict, **feed_dict},
                                     session=sess))
        return np.concatenate(emb)

    def classify(self, images, session, extra_feed_dict={}):
        """Compute logit scores for provided images."""
        return self.calc_embedding(images, self.test_logit, session, extra_feed_dict)

    def get_images(self, img_queue, lbl_queue, num_batches, sess):
        imgs = []
        lbls = []

        for i in range(int(num_batches)):
            i_, l_ = sess.run([img_queue, lbl_queue])
            imgs.append(i_)
            lbls.append(l_)

        images = np.vstack(imgs)
        labels = np.hstack(lbls)

        return images, labels

    def classify_using_embeddings(self, sup_imgs, sup_lbls, test_images,
                                  test_labels, sess):

        # if sup_imgs.shape:  # convert tensor to array
        #  sup_imgs, sup_lbls = self.get_images(sup_imgs, sup_lbls, 1, sess)

        sup_embs = self.calc_embedding(sup_imgs, self.test_emb, sess)
        test_embs = self.calc_embedding(test_images, self.test_emb, sess)

        match_ab = np.dot(sup_embs, np.transpose(test_embs))
        p_ba = softmax(np.transpose(match_ab))

        pred_ids = np.dot(p_ba, one_hot(sup_lbls, depth=self.num_labels))
        preds = np.argmax(pred_ids, axis=1)

        return preds  # np.mean(preds == test_labels)

    def calc_opt_logit_score(self, preds, lbls, k=None):
        # for the correct cluster score, we have to match clusters to classes
        # to do this, we can use the test labels to get the optimal matching
        # as in the literature, only the best clustering of all possible
        # clustering counts
        # CAUTION: this is only an upper bound on the accuracy - multiple
        # clusters can be assigned the same label
        #   (the resulting confusion matrix can have empty columns)
        #   typically only happens in cases of low accuracy
        #   -> use calc_correct_logit_score instead

        if k is None:
            k = self.num_labels

        pred_map = np.ones(k, np.int) * -1

        for i in range(k):
            samples_with_pred_i = (preds == i)
            labels = np.bincount(lbls[samples_with_pred_i])
            if len(labels) > 0:
                # labels[pred_map[pred_map > -1]] = -1
                pred_map[i] = labels.argmax()

        # classify with closest sample
        preds = pred_map[preds]

        conf_mtx = confusion_matrix(lbls, preds, self.num_labels)

        return conf_mtx, np.mean(preds == lbls)

    def train_and_eval_svm(self, train_images, train_labels, test_images,
                           test_labels, sess, num_samples=5000):

        if len(train_labels) < num_samples:
            print('less labels')
            num_samples = len(train_labels)
        # train svm
        X = self.calc_embedding(train_images[:num_samples], self.test_emb, sess)
        y = train_labels[:num_samples]

        clf = svm.SVC()
        clf.fit(X, y)

        y_train_pred = clf.predict(X)
        train_accuracy = np.mean(y_train_pred == y)

        test_embs = self.calc_embedding(test_images, self.test_emb, sess)

        y_t = clf.predict(test_embs)
        test_accuracy = np.mean(y_t == test_labels)

        return test_accuracy, train_accuracy

    def train_and_eval_svm_on_preds(self, train_preds, train_labels, test_preds,
                           test_labels, sess, num_samples=5000):

        if len(train_labels) < num_samples:
            print('less labels')
            num_samples = len(train_labels)
        # train svm
        X = train_preds[:num_samples]
        y = train_labels[:num_samples]

        clf = svm.SVC()
        clf.fit(X, y)

        y_train_pred = clf.predict(X)
        train_accuracy = np.mean(y_train_pred == y)

        y_t = clf.predict(test_preds)
        test_accuracy = np.mean(y_t == test_labels)

        return test_accuracy, train_accuracy

    def add_sat_loss(self, t_unsup_embs_logits, t_aug_embs_logits, weight=1.0):
        """loss as in IMSAT paper"""

        softmaxed_unsup = tf.nn.softmax(t_unsup_embs_logits)
        print('sat')

        self.sat_loss = tf.losses.softmax_cross_entropy(
          softmaxed_unsup,
          t_aug_embs_logits,
          scope='loss_sat',
          weights=weight,
          loss_collection=NO_FC_COLLECTION)
        tf.summary.scalar('Loss_SAT', self.sat_loss)

        return self.sat_loss

    def add_transformation_loss(self, t_embs, t_aug_embs, t_embs_logits,
                                t_aug_embs_logits, batch_size, weight=1, label_smoothing=0, loss_collection=LOSSES_COLLECTION):
        """ Add a transformation loss.
        Args:
            t_embs: embeddings of input images
            t_aug_embs: embeddings of augmented input images
            t_embs_logits: logits of input images (pre-softmax!)
            t_aug_embs_logits: logits of augmented input images (pre-softmax!)
        Returns:
            Transformation loss.
        """
        t_all_embs = tf.concat([t_embs, t_aug_embs], axis=0)

        t_all_logits = tf.concat([t_embs_logits, t_aug_embs_logits], axis=0)
        t_all_logits_softmaxed = tf.nn.softmax(t_all_logits)

        batch_size *= 2  # due to concatenation if batch_size is the same for embs and aug_embs
        t_emb_sim = tf.matmul(t_all_embs, t_all_embs, transpose_b=True,
                              name='emb_similarity')
        t_emb_sim = tf.reshape(t_emb_sim, [batch_size ** 2])

        # TODO(haeusser) use xentropy without softmax
        t_xentropy = tf.losses.softmax_cross_entropy(
                tf_repeat(t_all_logits_softmaxed, [batch_size, 1]),  # "labels"
                tf.tile(t_all_logits, [batch_size, 1]),  # will be softmaxed
                label_smoothing=label_smoothing,
                loss_collection=None,  # this is not the final loss yet
                )

        t_target = tf.ones([batch_size ** 2]) - tf.cast(t_xentropy, dtype=tf.float32)

        self.t_transf_loss = tf.reduce_mean(tf.abs(t_target - t_emb_sim)) * weight
        tf.add_to_collection(loss_collection, self.t_transf_loss)

        tf.summary.scalar('Loss_Transf', self.t_transf_loss)

        return self.t_transf_loss
    def add_transformation_loss_sparse(self, t_embs, t_aug_embs, t_embs_logits,
                                t_aug_embs_logits, num_embs, num_aug_embs, weight=1, label_smoothing=0, loss_collection=LOSSES_COLLECTION):
        """ Add a transformation loss.
        Args:
            t_embs: embeddings of input images
            t_aug_embs: embeddings of augmented input images
            t_embs_logits: logits of input images (pre-softmax!)
            t_aug_embs_logits: logits of augmented input images (pre-softmax!)
        Returns:
            Transformation loss. Does not compute xentropy in same class
        """

        t_logits_softmaxed = tf.nn.softmax(t_embs_logits)
        #t_aug_logits_softmaxed = tf.nn.softmax(t_aug_embs_logits)

        t_emb_sim = tf.matmul(t_embs, t_aug_embs, transpose_b=True,
                              name='emb_similarity')
        t_emb_sim = tf.reshape(t_emb_sim, [num_embs * num_aug_embs])

        # TODO(haeusser) use xentropy without softmax
        t_xentropy = tf.losses.softmax_cross_entropy(
                tf_repeat(t_logits_softmaxed, [num_aug_embs, 1]),  # "labels"
                tf.tile(t_aug_embs_logits, [num_embs, 1]),  # will be softmaxed
                label_smoothing=label_smoothing,
                loss_collection=None,  # this is not the final loss yet
                )

        t_target = tf.ones([num_embs * num_aug_embs]) - tf.cast(t_xentropy, dtype=tf.float32)

        self.t_transf_loss = tf.reduce_mean(tf.abs(t_target - t_emb_sim)) * weight
        tf.add_to_collection(loss_collection, self.t_transf_loss)

        tf.summary.scalar('Loss_Transf_sp', self.t_transf_loss)

        return self.t_transf_loss



def do_kmeans(embs, lbls, num_labels):
    kmeans = KMeans(n_clusters=num_labels, random_state=0).fit(embs)
    preds = kmeans.labels_
    conf_mtx, score = calc_correct_logit_score(preds, lbls, num_labels)

    return conf_mtx, score


def calc_correct_logit_score(preds, lbls, num_labels):
    # for the correct cluster score, a one-to-one mapping of clusters to
    # classes is necessary
    # this can be done using the hungarian algorithm
    # (it is not allowed to assign two clusters to the same class label,
    # even if that would benefit overall score

    conf_mtx = confusion_matrix(lbls, preds, num_labels)

    assi = linear_sum_assignment(-conf_mtx)

    acc = conf_mtx[assi].sum() / conf_mtx.sum()

    return conf_mtx[:, assi[1]], acc


def calc_nmi(preds, lbls):
    nmi = normalized_mutual_info_score(preds, lbls)
    return nmi

def calc_sat_score(unsup_emb, reg_unsup_emb):
    "accuracy for self augmentation (embedding augmented image close to non-augmented image)"
    proximity = np.dot(reg_unsup_emb, unsup_emb.T)
    closest_unsup = proximity.argmax(-1)
    should_be = np.repeat(np.arange(len(unsup_emb)), len(reg_unsup_emb) // len(unsup_emb))
    acc = np.mean(closest_unsup == should_be)
    return acc
