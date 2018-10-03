#! /usr/bin/env python
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

Association-based semi-supervised training example in MNIST dataset.

Training should reach ~1% error rate on the test set using 100 labeled samples
in 5000-10000 steps (a few minutes on Titan X GPU)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 2,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,  #-1 -> choose randomly   -2 -> use sup_per_class as seed
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', -1,   #-1 -> take all available
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')
flags.DEFINE_float('logit_weight', 1.0, 'Weight for logits')
flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout factor.')
flags.DEFINE_float('l1_weight', 0.0002, 'Weight for l1 embeddding regularization')

flags.DEFINE_integer('warmup_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

flags.DEFINE_bool('semisup', True, 'Add unsupervised samples')

print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

from tools import mnist as mnist_tools
import numpy as np

NUM_LABELS = mnist_tools.NUM_LABELS
IMAGE_SHAPE = mnist_tools.IMAGE_SHAPE


def main(_):
    train_images, train_labels = mnist_tools.get_data('train')
    test_images, test_labels = mnist_tools.get_data('test')

    # Sample labeled training subset.
    if FLAGS.sup_seed >= 0:
      seed = FLAGS.sup_seed
    elif FLAGS.sup_seed == -2:
      seed = FLAGS.sup_per_class
    else:
      seed = np.random.randint(0, 1000)

    print('Seed:', seed)
    sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                           FLAGS.sup_per_class, NUM_LABELS, seed)


    graph = tf.Graph()
    with graph.as_default():
        model = semisup.SemisupModel(semisup.architectures.mnist_model, NUM_LABELS,
                                     IMAGE_SHAPE, dropout_keep_prob=FLAGS.dropout_keep_prob)

        # Set up inputs.
        t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
                    sup_by_label, FLAGS.sup_per_batch)

        # Compute embeddings and logits.
        t_sup_emb = model.image_to_embedding(t_sup_images)
        t_sup_logit = model.embedding_to_logit(t_sup_emb)

        # Add losses.
        if FLAGS.semisup:
            t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                                         FLAGS.unsup_batch_size)

            t_unsup_emb = model.image_to_embedding(t_unsup_images)
            model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels,
                    walker_weight=FLAGS.walker_weight, visit_weight=FLAGS.visit_weight)

            #model.add_emb_regularization(t_unsup_emb, weight=FLAGS.l1_weight)

        model.add_logit_loss(t_sup_logit, t_sup_labels, weight=FLAGS.logit_weight)

        #model.add_emb_regularization(t_sup_emb, weight=FLAGS.l1_weight)

        t_learning_rate = tf.placeholder("float", shape=[])

        train_op = model.create_train_op(t_learning_rate)


    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        learning_rate_ = FLAGS.learning_rate

        for step in range(FLAGS.max_steps):
            lr = learning_rate_
            if step < FLAGS.warmup_steps:
                lr = 1e-6 + semisup.apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)

            _ = sess.run([train_op], {
              t_learning_rate: lr
            })
            if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
                print('Step: %d' % step)
                test_pred = model.classify(test_images, sess).argmax(-1)
                conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
                test_err = (test_labels != test_pred).mean() * 100
                print(conf_mtx)
                print('Test error: %.2f %%' % test_err)
                print()


            if step % FLAGS.decay_steps == 0 and step > 0:
                learning_rate_ = learning_rate_ * FLAGS.decay_factor


        coord.request_stop()
        coord.join(threads)

    print('FINAL RESULTS:')
    print('Test error: %.2f %%' % (test_err))
    print('final_score', 1 - test_err/100)

    print('@@test_error:%.4f' % (test_err/100))
    print('@@train_loss:%.4f' % 0)
    print('@@reg_loss:%.4f' % 0)
    print('@@estimated_error:%.4f' % 0)
    print('@@centroid_norm:%.4f' % 0)
    print('@@emb_norm:%.4f' % 0)
    print('@@k_score:%.4f' % 0)
    print('@@svm_score:%.4f' % 0)


if __name__ == '__main__':
    app.run()
