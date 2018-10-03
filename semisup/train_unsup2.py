#! /usr/bin/env python
"""
Association based clustering on STL10.

Uses a second association based loss for regularization:
  Augmented samples should be associated to non-augmented samples.
  This prevents the algorithm from finding 'too easy' and 'useless' clusters

Runs on mnist, cifar10, svhn, stl10 and potentially more datasets


usage:
   python3 train_unsup2.py [args]

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from importlib import import_module

FLAGS = flags.FLAGS

flags.DEFINE_integer('virtual_embeddings_per_class', 4,
                     'Number of image centroids per class')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 1000,
                     'Number of steps between evaluations.')

flags.DEFINE_integer('svm_test_interval', 10000, 'Number of steps between SVM evaluations.')
flags.DEFINE_float('learning_rate', 2e-4, 'Initial learning rate.')

flags.DEFINE_integer('warmup_steps', 1000, 'Warmup steps.')
flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')
flags.DEFINE_float('decay_steps', 5000, 'Learning rate decay interval in steps.')

flags.DEFINE_integer('reg_warmup_steps', 1, 'Warmup steps for regularization walker.')
flags.DEFINE_integer('num_unlabeled_images', 0, 'How many images to use from the unlabeled set.')

flags.DEFINE_float('visit_weight_base', 0.5, 'Weight for visit loss.')
flags.DEFINE_float('rvisit_weight', 1, 'Weight for reg visit loss.')
flags.DEFINE_float('reg_decay_factor', 0.2, 'Decay reg weight after kmeans initialization')

flags.DEFINE_float('cluster_association_weight', 1.0, 'Weight for cluster associations.')
flags.DEFINE_float('reg_association_weight', 1.0, 'Weight for reg associations.')
flags.DEFINE_float('trafo_weight', 0, 'Weight for 4) transformation loss.')

flags.DEFINE_float('beta1', 0.8, 'beta1 parameter for adam')
flags.DEFINE_float('beta2', 0.9, 'beta2 parameter for adam')

flags.DEFINE_float('l1_weight', 0.0002, 'Weight for l1 embeddding regularization')
flags.DEFINE_float('norm_weight', 0.0002, 'Weight for embedding normalization')
flags.DEFINE_float('logit_weight', 0.5, 'Weight for logit loss')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')
flags.DEFINE_float('rwalker_weight', 1.0, 'Weight for reg walker loss.')
flags.DEFINE_bool('normalize_input', True, 'Normalize input images to be between -1 and 1. Requires tanh autoencoder')

flags.DEFINE_integer('max_steps', 200000, 'Number of training steps.')
flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')
flags.DEFINE_integer('taskid', None, 'Id of current task. Will be added to logdir')
flags.DEFINE_integer('num_blocks', 3, 'Number of blocks in resnet')
flags.DEFINE_integer('num_augmented_samples', 3, 'Number of augmented samples for each image.')
flags.DEFINE_float('scale_match_ab', 1,
                   'How to scale match ab to prevent numeric instability. Use when using embedding normalization')
flags.DEFINE_float('norm_target', 1, 'Target of embedding normalization')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer. Can be [adam, sgd, rms]')

flags.DEFINE_string('init_method', 'normal_center03',
                    'How to initialize image centroids. Should be one  of [uniform_128, uniform_10, uniform_255, avg, '
                    'random_center]')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')
flags.DEFINE_float('zero_fact', 1, 'Used for simulation imbalanced class distribution. Only this fraction of zeros will be used.')

flags.DEFINE_string('logdir', None, 'Where to put the logs. By default, no logs will be saved.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to work on.')
flags.DEFINE_string('architecture', 'mnist_model_dropout', 'Which network architecture '
                                                           'from architectures.py to use.' )

flags.DEFINE_string('restore_checkpoint', None, 'restore weights from checkpoint, e.g. some autoencoder pretraining')
flags.DEFINE_bool('init_with_kmeans', False, 'Initialize centroids using kmeans after reg_warmup_steps steps.')
flags.DEFINE_bool('normalize_embeddings', False, 'Normalize embeddings (l2 norm = 1)')
flags.DEFINE_bool('volta', False, 'Use more CPU for preprocessing to load GPU')
flags.DEFINE_bool('use_test', False, 'Use Test images as part of training set. Done by a few clustering algorithms')
flags.DEFINE_float('kmeans_sat_thresh', None, 'Init with kmeans when SAT accuracy > x')
flags.DEFINE_bool('trafo_separate_loss_collection', False, 'Do ignore gradients for transformation loss on last fc layer')
flags.DEFINE_bool('shuffle_augmented_samples', False,
                  'If true, the augmented samples are shuffled separately. Otherwise, a batch contains augmentated '
                  'samples of its non-augmented samples')

print(FLAGS.learning_rate)
print("flags:", str(FLAGS.__flags))  # print all flags (useful when logging)

import numpy as np

np.core.arrayprint._line_width = 150
from semisup.backend import apply_envelope
from backend import apply_envelope
import semisup
from tensorflow.contrib.data import Dataset
from augment import apply_augmentation


def main(_):
    FLAGS.eval_interval = 1000  # todo remove
    if FLAGS.logdir is not None:
        if FLAGS.taskid is not None:
            FLAGS.logdir = FLAGS.logdir + '/t_' + str(FLAGS.taskid)
        else:
            FLAGS.logdir = FLAGS.logdir + '/t_' + str(random.randint(0,99999))

    dataset_tools = import_module('tools.' + FLAGS.dataset)

    NUM_LABELS = dataset_tools.NUM_LABELS
    num_labels = NUM_LABELS
    IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
    image_shape = IMAGE_SHAPE

    train_images, train_labels_svm = dataset_tools.get_data('train')  # no train labels nowhere
    test_images, test_labels = dataset_tools.get_data('test')

    if FLAGS.zero_fact < 1:
        # exclude a random set of zeros (not at the end, then there would be many batches without zeros)
        keep = np.ones(len(train_labels_svm), dtype=bool)
        zero_indices = np.where((train_labels_svm == 0))[0]

        remove = np.random.uniform(0, 1, len(zero_indices))
        zero_indices_to_remove = zero_indices[remove > FLAGS.zero_fact]

        keep[zero_indices_to_remove] = False

        train_images = train_images[keep]
        train_labels_svm = train_labels_svm[keep]

        print('using only a fraction of zeros, resulting in the following shape:', train_images.shape)

    if FLAGS.num_unlabeled_images > 0:
        unlabeled_train_images, _ = dataset_tools.get_data('unlabeled', max_num=np.min([FLAGS.num_unlabeled_images, 50000]))
        train_images = np.vstack([train_images, unlabeled_train_images])

    if FLAGS.normalize_input:
        train_images = (train_images - 128.) / 128.
        test_images = (test_images - 128.) / 128.

    if FLAGS.use_test:
        train_images = np.vstack([train_images, test_images])
        train_labels_svm = np.hstack([train_labels_svm, test_labels])

    #if FLAGS.dataset == 'svhn' and FLAGS.architecture == 'resnet_cifar_model':
    #  FLAGS.emb_size = 64

    image_shape_crop = image_shape
    c_test_imgs = test_images
    c_train_imgs = train_images

    # crop images to some random region. Intuitively, images should belong to the same cluster,
    # even if a part of the image is missing
    # (no padding, because the net could detect padding easily, and match it to other augmented samples that have
    # padding)
    if FLAGS.dataset == 'stl10':
        image_shape_crop = [64, 64, 3]
        c_test_imgs = test_images[:, 16:80, 16:80]
        c_train_imgs = train_images[:, 16:80, 16:80]

    def aug(image):
        return apply_augmentation(image, target_shape=image_shape_crop, params=dataset_tools.augmentation_params)

    def random_crop(image):
        image_size = image_shape_crop[0]
        image = tf.random_crop(image, [image_size, image_size, image_shape[2]])

        return image

    graph = tf.Graph()
    with graph.as_default():
        t_images = tf.placeholder("float", shape=[None] + image_shape)

        dataset = Dataset.from_tensor_slices(t_images)
        dataset = dataset.shuffle(buffer_size=10000, seed=47)  # important, so that we have the same images in both sets

        # parameters for buffering during augmentation. Only influence training speed.
        nt = 8 if FLAGS.volta else 4    # that's not even enough, but there are no more CPUs
        b = 10000

        rf = FLAGS.num_augmented_samples

        augmented_set = dataset
        if FLAGS.shuffle_augmented_samples:
            augmented_set = augmented_set.shuffle(buffer_size=10000, seed=47)

        # get multiple augmented versions of the same image - they should later have similar embeddings
        augmented_set = augmented_set.flat_map(lambda x: Dataset.from_tensors(x).repeat(rf))

        augmented_set = augmented_set.map(aug, num_threads=nt, output_buffer_size=b)

        dataset = dataset.map(random_crop, num_threads=1, output_buffer_size=b)
        dataset = dataset.repeat().batch(FLAGS.unsup_batch_size)
        augmented_set = augmented_set.repeat().batch(FLAGS.unsup_batch_size * rf)

        iterator = dataset.make_initializable_iterator()
        reg_iterator = augmented_set.make_initializable_iterator()

        t_unsup_images = iterator.get_next()
        t_reg_unsup_images = reg_iterator.get_next()

        model_func = getattr(semisup.architectures, FLAGS.architecture)

        model = semisup.SemisupModel(model_func, num_labels, image_shape_crop, optimizer='adam',
                                     emb_size=FLAGS.emb_size,
                                     dropout_keep_prob=FLAGS.dropout_keep_prob, num_blocks=FLAGS.num_blocks,
                                     normalize_embeddings=FLAGS.normalize_embeddings, beta1=FLAGS.beta1,
                                     beta2=FLAGS.beta2)

        init_virt = []
        for c in range(num_labels):
            center = np.random.normal(0, 0.3, size=[1, FLAGS.emb_size])
            noise = np.random.uniform(-0.01, 0.01, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
            centroids = noise + center
            init_virt.extend(centroids)

        t_sup_emb = tf.Variable(tf.cast(np.array(init_virt), tf.float32), name="virtual_centroids")

        t_sup_labels = tf.constant(
            np.concatenate([[i] * FLAGS.virtual_embeddings_per_class for i in range(num_labels)]))

        visit_weight = tf.placeholder("float", shape=[])
        walker_weight = tf.placeholder("float", shape=[])
        t_logit_weight = tf.placeholder("float", shape=[])
        t_trafo_weight = tf.placeholder("float", shape=[])

        t_l1_weight = tf.placeholder("float", shape=[])
        t_norm_weight = tf.placeholder("float", shape=[])
        t_learning_rate = tf.placeholder("float", shape=[])
        t_sat_loss_weight = tf.placeholder("float", shape=[])

        t_unsup_emb = model.image_to_embedding(t_unsup_images)
        t_reg_unsup_emb = model.image_to_embedding(t_reg_unsup_images)

        t_all_unsup_emb = tf.concat([t_unsup_emb, t_reg_unsup_emb], axis=0)
        t_rsup_labels = tf.constant(np.concatenate([[i] * rf for i in range(FLAGS.unsup_batch_size)]))

        rwalker_weight = tf.placeholder("float", shape=[])
        rvisit_weight = tf.placeholder("float", shape=[])

        if FLAGS.normalize_embeddings:
            t_sup_logit = model.embedding_to_logit(tf.nn.l2_normalize(t_sup_emb, dim=1))
            model.add_semisup_loss(
                    tf.nn.l2_normalize(t_sup_emb, dim=1), tf.nn.l2_normalize(t_unsup_emb, dim=1), t_sup_labels,
                    walker_weight=walker_weight, visit_weight=visit_weight,
                    match_scale=FLAGS.scale_match_ab)
            model.reg_loss_aba = model.add_semisup_loss(
                    tf.nn.l2_normalize(t_reg_unsup_emb, dim=1), tf.nn.l2_normalize(t_unsup_emb, dim=1), t_rsup_labels,
                    walker_weight=rwalker_weight, visit_weight=rvisit_weight, match_scale=FLAGS.scale_match_ab,
                    est_err=False)

        else:
            t_sup_logit = model.embedding_to_logit(t_sup_emb)
            model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels,
                    walker_weight=walker_weight, visit_weight=visit_weight,
                    match_scale=FLAGS.scale_match_ab, est_err=True, name='c_association')
            model.reg_loss_aba = model.add_semisup_loss(
                    t_reg_unsup_emb, t_unsup_emb, t_rsup_labels,
                    walker_weight=rwalker_weight, visit_weight=rvisit_weight, match_scale=FLAGS.scale_match_ab, est_err=False, name='aug_association')

        model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

        t_reg_unsup_emb_singled = t_reg_unsup_emb[::FLAGS.num_augmented_samples]

        t_unsup_logit = model.embedding_to_logit(t_unsup_emb)
        t_reg_unsup_logit = model.embedding_to_logit(t_reg_unsup_emb_singled)

        model.add_sat_loss(t_unsup_logit, t_reg_unsup_logit, weight=t_sat_loss_weight)

        trafo_lc = semisup.NO_FC_COLLECTION if FLAGS.trafo_separate_loss_collection else semisup.LOSSES_COLLECTION

        if FLAGS.trafo_weight > 0:
          # only use a single augmented sample per sample

          t_trafo_loss = model.add_transformation_loss(t_unsup_emb, t_reg_unsup_emb_singled, t_unsup_logit,
                                                     t_reg_unsup_logit, FLAGS.unsup_batch_size, weight=t_trafo_weight, label_smoothing=0, loss_collection=trafo_lc)
        else:
          t_trafo_loss = tf.constant(0)

        model.add_emb_regularization(t_all_unsup_emb, weight=t_l1_weight)
        model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)

        # make l2 norm = 3
        model.add_emb_normalization(t_sup_emb, weight=t_norm_weight, target=FLAGS.norm_target)
        model.add_emb_normalization(t_all_unsup_emb, weight=t_norm_weight, target=FLAGS.norm_target)

        gradient_multipliers = {t_sup_emb: 1 }
        [train_op, train_op_sat] = model.create_train_op(t_learning_rate, gradient_multipliers=gradient_multipliers)

        summary_op = tf.summary.merge_all()
        if FLAGS.logdir is not None:
            summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)
            saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        sess.run(iterator.initializer, feed_dict={t_images: train_images})
        sess.run(reg_iterator.initializer, feed_dict={t_images: train_images})

        # optional: init from autoencoder
        if FLAGS.restore_checkpoint is not None:
            # logit fc layer cannot be restored
            def is_main_net(x):
                return 'logit_fc' not in x.name and 'Adam' not in x.name

            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
            variables = list(filter(is_main_net, variables))

            restorer = tf.train.Saver(var_list=variables)
            restorer.restore(sess, FLAGS.restore_checkpoint)

        extra_feed_dict = {}

        from numpy.linalg import norm

        reg_warmup_steps = FLAGS.reg_warmup_steps
        logit_weight_ = FLAGS.logit_weight
        rwalker_weight_ = FLAGS.rwalker_weight
        rvisit_weight_ = FLAGS.rvisit_weight
        learning_rate_ = FLAGS.learning_rate
        trafo_weight = FLAGS.trafo_weight

        kmeans_initialized = False

        for step in range(FLAGS.max_steps):
            import time
            start = time.time()
            if FLAGS.init_with_kmeans:
                if FLAGS.kmeans_sat_thresh is not None and not kmeans_initialized or \
                        FLAGS.kmeans_sat_thresh is None and step <= reg_warmup_steps:
                    walker_weight_ = 0
                    visit_weight_ = 0
                    logit_weight_ = 0
                    trafo_weight = 0
                else:
                    walker_weight_ = FLAGS.walker_weight
                    visit_weight_ = FLAGS.visit_weight_base
                    logit_weight_ = FLAGS.logit_weight
                    trafo_weight = FLAGS.trafo_weight
            else:
                walker_weight_ = apply_envelope("log", step, FLAGS.walker_weight, reg_warmup_steps, 0)
                visit_weight_ = apply_envelope("log", step, FLAGS.visit_weight_base, reg_warmup_steps, 0)

            feed_dict = {rwalker_weight: rwalker_weight_ * FLAGS.reg_association_weight,
                         rvisit_weight: rvisit_weight_ * FLAGS.reg_association_weight,
                         walker_weight: walker_weight_ * FLAGS.cluster_association_weight,
                         visit_weight: visit_weight_ * FLAGS.cluster_association_weight,
                         t_l1_weight: FLAGS.l1_weight,
                         t_norm_weight: FLAGS.norm_weight,
                         t_logit_weight: logit_weight_,
                         t_trafo_weight: trafo_weight,
                         t_sat_loss_weight: 0,
                         t_learning_rate: 1e-6 + apply_envelope("log", step, learning_rate_, FLAGS.warmup_steps, 0)
            }
            _, sat_loss, train_loss, summaries, centroids, unsup_emb, reg_unsup_emb, estimated_error, p_ab, p_ba, p_aba, \
            reg_loss, trafo_loss = sess.run(
                    [train_op, train_op_sat, model.train_loss, summary_op, t_sup_emb, t_unsup_emb, t_reg_unsup_emb,
                     model.estimate_error, model.p_ab,
                     model.p_ba, model.p_aba, model.reg_loss_aba, t_trafo_loss], {**extra_feed_dict, **feed_dict})

            if FLAGS.kmeans_sat_thresh is not None and step % 200 == 0 and not kmeans_initialized:
                sat_score = semisup.calc_sat_score(unsup_emb, reg_unsup_emb)

                if sat_score > FLAGS.kmeans_sat_thresh:
                    print('initializing with kmeans', step, sat_score)
                    FLAGS.init_with_kmeans = True
                    kmeans_initialized = True
                    reg_warmup_steps = step # -> jump to next if clause

            if FLAGS.init_with_kmeans and step == reg_warmup_steps:
                # do kmeans, initialize with kmeans
                embs = model.calc_embedding(c_train_imgs, model.test_emb, sess, extra_feed_dict)

                kmeans = semisup.KMeans(n_clusters=num_labels, random_state=0).fit(embs)

                init_virt = []
                noise = 0.0001
                for c in range(num_labels):
                    center = kmeans.cluster_centers_[c]
                    noise = np.random.uniform(-noise, noise, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
                    centroids = noise + center
                    init_virt.extend(centroids)

                # init with K-Means
                assign_op = t_sup_emb.assign(np.array(init_virt))
                sess.run(assign_op)
                model.reset_optimizer(sess)

                rwalker_weight_ *= FLAGS.reg_decay_factor
                rvisit_weight_ *= FLAGS.reg_decay_factor

            if FLAGS.svm_test_interval is not None and step % FLAGS.svm_test_interval == 0 and step > 0:
                svm_test_score, _ = model.train_and_eval_svm(c_train_imgs, train_labels_svm, c_test_imgs, test_labels,
                                                             sess, num_samples=5000)
                print('svm score:', svm_test_score)
                test_pred = model.classify(c_test_imgs, sess)
                train_pred = model.classify(c_train_imgs, sess)
                svm_test_score, _ = model.train_and_eval_svm_on_preds(train_pred, train_labels_svm, test_pred, test_labels,
                                                             sess, num_samples=5000)
                print('svm score on logits:', svm_test_score)

            if step % FLAGS.decay_steps == 0 and step > 0:
                learning_rate_ = learning_rate_ * FLAGS.decay_factor

            if step == 0 or (step + 1) % FLAGS.eval_interval == 0 or step == 99:
                print('Step: %d' % step)
                print('trafo loss', trafo_loss)
                print('reg loss' , reg_loss)
                print('Time for step', time.time() - start)
                test_pred = model.classify(c_test_imgs, sess, extra_feed_dict).argmax(-1)

                nmi = semisup.calc_nmi(test_pred, test_labels)

                conf_mtx, score = semisup.calc_correct_logit_score(test_pred, test_labels, num_labels)
                print(conf_mtx)
                print('Test error: %.2f %%' % (100 - score * 100))
                print('Test NMI: %.2f %%' % (nmi * 100))
                print('Train loss: %.2f ' % train_loss)
                print('Train loss no fc: %.2f ' % sat_loss)
                print('Reg loss aba: %.2f ' % reg_loss)
                print('Estimated Accuracy: %.2f ' % estimated_error)

                sat_score = semisup.calc_sat_score(unsup_emb, reg_unsup_emb)
                print('sat accuracy', sat_score)

                embs = model.calc_embedding(c_test_imgs, model.test_emb, sess, extra_feed_dict)

                c_n = norm(centroids, axis=1, ord=2)
                e_n = norm(embs[0:100], axis=1, ord=2)
                print('centroid norm', np.mean(c_n))
                print('embedding norm', np.mean(e_n))

                k_conf_mtx, k_score = semisup.do_kmeans(embs, test_labels, num_labels)
                print(k_conf_mtx)
                print('k means score:', k_score)  # sometimes that kmeans is better than the logits

                if FLAGS.logdir is not None:
                    sum_values = {
                        'test score': score,
                        'reg loss': reg_loss,
                        'centroid norm': np.mean(c_n),
                        'embedding norm': np.mean(c_n),
                        'k means score': k_score
                        }

                    summary_writer.add_summary(summaries, step)

                    for key, value in sum_values.items():
                        summary = tf.Summary(
                                value=[tf.Summary.Value(tag=key, simple_value=value)])
                        summary_writer.add_summary(summary, step)

                # early stopping to save some time
                if step == 34999 and score < 0.45:
                  break
                if step == 14999 and score < 0.225:
                  break

                if dataset == 'mnist' and step == 6999 and score < 0.25:
                  break


        svm_test_score, _ = model.train_and_eval_svm(c_train_imgs, train_labels_svm, c_test_imgs, test_labels, sess,
                                                     num_samples=10000)

        if FLAGS.logdir is not None:
            path = saver.save(sess, FLAGS.logdir, model.step)
            print('@@model_path:%s' % path)

        print('FINAL RESULTS:')
        print(conf_mtx)
        print('Test error: %.2f %%' % (100 - score * 100))
        print('final_score', score)

        print('@@test_error:%.4f' % score)
        print('@@train_loss:%.4f' % train_loss)
        print('@@reg_loss:%.4f' % reg_loss)
        print('@@estimated_error:%.4f' % estimated_error)
        print('@@centroid_norm:%.4f' % np.mean(c_n))
        print('@@emb_norm:%.4f' % np.mean(e_n))
        print('@@k_score:%.4f' % k_score)
        print('@@svm_score:%.4f' % svm_test_score)


if __name__ == '__main__':
    app.run()
