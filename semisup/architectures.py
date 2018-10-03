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

Definitions and utilities for the svhn model.

This file contains functions that are needed for semisup training and
evalutaion on the SVHN dataset.
They are used in svhn_train.py and svhn_eval.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from semisup.densenet import DenseNet


def densenet_model(inputs,
                   is_training=True,
                   emb_size=128,
                   l2_weight=1e-4,
                   img_shape=None,
                   new_shape=None,
                   image_summary=False,
                   augmentation_function=None,
                   num_blocks=None,
                   batch_norm_decay=0.99):  # pylint: disable=unused-argument
    """Construct the image-to-embedding vector model."""
    inputs = tf.cast(inputs, tf.float32)

    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)

    if is_training and augmentation_function is not None:
        tf.map_fn(lambda frame: augmentation_function(frame), inputs)

    if augmentation_function is not None:
        tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs)

    print(inputs.shape)
    shape = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
    densenet = DenseNet(data_shape=shape, n_classes=10, model_type='DenseNet', keep_prob=dropout_keep_prob,
                        growth_rate=12, depth=40, is_training=is_training)

    emb = densenet.build_embeddings(inputs)

    # with slim.arg_scope(
    #        [slim.fully_connected],
    #        activation_fn=tf.nn.elu,
    #        weights_regularizer=slim.l2_regularizer(l2_weight),
    #        normalizer_fn=None):

    #    emb = slim.fully_connected(emb, emb_size, scope='fc1')
    return emb


def resnet_cifar_model(inputs,
                       is_training=True,
                       emb_size=128,
                       l2_weight=1e-4,
                       img_shape=None,
                       new_shape=None,
                       image_summary=False,
                       augmentation_function=None,
                       num_blocks=None,
                       dropout_keep_prob=1,
                       batch_norm_decay=0.99,
                       resnet_size=20): # 3 residual blocks
    from official.resnet import resnet_model

    def _find_tensor(name, graph=tf.get_default_graph()):
        # graph is redefined multipletimes, use the last operation
        # this might break if operations are re-ordered for some reason
        a = None
        for op in graph.get_operations():
            if name in op.name:
                a = op.outputs[0]
        return a

    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)
    net = inputs
    network = resnet_model.cifar10_resnet_v2_generator(resnet_size, 1)
    logits = network(net, is_training)
    pre_emb = _find_tensor('final_avg_pool', logits.graph)
    emb = tf.contrib.slim.flatten(pre_emb)

    emb = slim.dropout(emb, dropout_keep_prob, scope='dropout')
    emb = tf.layers.dense(inputs=emb, units=emb_size)

    emb = tf.identity(emb, 'embeddings')

    return emb


from official.resnet.resnet_model import conv2d_fixed_padding, fixed_padding, bottleneck_block,building_block,block_layer,batch_norm_relu
def mnist_resnet_v2_generator(num_blocks=5, final_pool_size=8, dropout_keep_prob=1, data_format=None):
  """Generator for MNIST resnet model.
     This is ResNet30 for cifar - with a final avg pooling of size 7 instead of 8, as mnist images are slightly smaller (28px vs 32px)

  Args:
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  n_filters = 16

  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=n_filters, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = block_layer(
        inputs=inputs, filters=n_filters, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    if dropout_keep_prob < 1:
        inputs = slim.dropout(inputs, dropout_keep_prob, scope='dropout')


    inputs = block_layer(
        inputs=inputs, filters=n_filters*2, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    if dropout_keep_prob < 1:
        inputs = slim.dropout(inputs, dropout_keep_prob, scope='dropout')

    inputs = block_layer(
        inputs=inputs, filters=n_filters*4, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=int(final_pool_size), strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    return inputs

  return model

def stl10_resnet_v2_generator(num_blocks=2, data_format=None):
  """Generator for STL-10 ResNet v2 models.
  adapted according to https://128.84.21.199/pdf/1707.07103.pdf

  Args:
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=32, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=2, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=32, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    return inputs

  return model

def resnet_stl10_model(inputs,
                       is_training=True,
                       emb_size=128,
                       l2_weight=1e-4,
                       img_shape=None,
                       new_shape=None,
                       image_summary=False,
                       augmentation_function=None,
                       num_blocks=2,
                       dropout_keep_prob=1,
                       batch_norm_decay=0.99,
                       resnet_size=32):

    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)

    net = inputs
    network = stl10_resnet_v2_generator(num_blocks=num_blocks)
    net = network(net, is_training)
    net = tf.contrib.slim.flatten(net)
    net = slim.dropout(net, dropout_keep_prob, scope='dropout')
    emb = tf.layers.dense(inputs=net, units=emb_size)

    emb = tf.identity(emb, 'embeddings')
    return emb

def resnet_mnist_model(inputs,
                       is_training=True,
                       emb_size=128,
                       l2_weight=1e-4,
                       img_shape=None,
                       new_shape=None,
                       image_summary=False,
                       augmentation_function=None,
                       num_blocks=2,
                       dropout_keep_prob=1,
                       batch_norm_decay=0.99,
                       resnet_size=32):

    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)

    net = inputs
    network = mnist_resnet_v2_generator(num_blocks, final_pool_size=inputs.shape[1]//4)
    net = network(net, is_training)
    net = tf.contrib.slim.flatten(net)
    net = slim.dropout(net, dropout_keep_prob, scope='dropout')
    net = tf.layers.dense(inputs=net, units=emb_size)

    net = tf.identity(net, 'embeddings')
    return net

def resnet_mnist_model_d(inputs,
                       is_training=True,
                       emb_size=128,
                       l2_weight=1e-4,
                       img_shape=None,
                       new_shape=None,
                       image_summary=False,
                       augmentation_function=None,
                       num_blocks=None,
                       dropout_keep_prob=1,
                       batch_norm_decay=0.99,
                       resnet_size=32):

    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)

    net = inputs
    network = mnist_resnet_v2_generator(num_blocks, final_pool_size=7, dropout_keep_prob=dropout_keep_prob)
    net = network(net, is_training)
    net = tf.contrib.slim.flatten(net)
    net = tf.layers.dense(inputs=net, units=emb_size)

    net = tf.identity(net, 'embeddings')
    return net


def stl10_ex_cnn(inputs,
                 is_training=True,
                 augmentation_function=None,
                 emb_size=1024,
                 img_shape=None,
                 new_shape=None,
                 dropout_keep_prob=None,
                 num_blocks=None,
                 image_summary=False,
                 batch_norm_decay=0.99):
    """Construct the image-to-embedding model."""
    inputs = tf.cast(inputs, tf.float32)

    net = inputs


    with slim.arg_scope([slim.dropout], is_training=is_training):
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(5e-4), ):  # todo check l2
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                # 64
                print('start', net.shape)
                net = fixed_padding(net, kernel_size=5, data_format='channels_last')
                print('pad1', net.shape)
                net = slim.conv2d(net, 92, [5, 5], padding='VALID', scope='conv1')  #
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  #
                print(net.shape)

                net = fixed_padding(net, kernel_size=5, data_format='channels_last')
                print('pad2', net.shape)

                net = slim.conv2d(net, 256, [5, 5], padding='VALID', scope='conv2')  #
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')  #
                print(net.shape)

                net = fixed_padding(net, kernel_size=5, data_format='channels_last')
                print('pad3', net.shape)

                net = slim.conv2d(net, 512, [5, 5], padding='VALID', scope='conv3')  #
                #net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')  #
                print(net.shape)

                # todo this should not be necessary
                net = fixed_padding(net, kernel_size=2, data_format='channels_last')
                print('pad4', net.shape)
                net = slim.conv2d(net, 1024, [8, 8], padding='VALID', scope='conv4')  #
                net = slim.flatten(net, scope='flatten')
                print('fl',net.shape)

                with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                  emb = slim.fully_connected(
                    net, emb_size, activation_fn=None, scope='fc1')
                    # net = slim.dropout(net, dropout_keep_prob,  scope='dropout3')

    return emb

# A Resnet architecture from keras, see https://keras.io/applications/#resnet50
# Can be used with the standard weights from He et al.
# train_unsup2.py for an example
# note: when using tf.global_var_intializer, the pretrained weights have to be loaded again
# sets the global k_endpoint variable with the 2048-dimensional embedding vectors, useful for debugging
# (e.g. for calculating k-means on the resnet embeddings)
# resizes images to imagenet size, and applies standard preprocessing

# if num_blocks is 0, gradients are not propagated through the net (only the embedding layer will be trained)

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
k_model = None
k_endpoint = None

def keras_resnet(inputs,
                 is_training=True,
                 emb_size=128,
                 l2_weight=1e-4,
                 img_shape=None,
                 new_shape=None,
                 image_summary=False,
                 augmentation_function=None,
                 num_blocks=None,
                 dropout_keep_prob=1,
                 batch_norm_decay=0.99,
                 resnet_size=32):
    global k_endpoint
    global k_model

    if k_model is None:
        k_model = ResNet50(include_top=False)

    inputs = tf.cast(inputs, tf.float32)
    resized = tf.image.resize_images(inputs, (224, 224))
    inputs = preprocess_input(resized)

    resnet = k_model(inputs)
    if k_endpoint is None:
        k_endpoint = resnet

    net = tf.contrib.slim.flatten(resnet)

    if num_blocks == 0:
        net = tf.stop_gradient(net)
    net = tf.layers.dense(inputs=net, units=emb_size)

    net = tf.identity(net, 'embeddings')
    return net


def svhn_model(inputs,
               is_training=True,
               augmentation_function=None,
               emb_size=128,
               l2_weight=1e-4,
               img_shape=None,
               new_shape=None,
               image_summary=False,
               num_blocks=None,
               dropout_keep_prob=1,
               batch_norm_decay=0.99):  # pylint: disable=unused-argument
    """Construct the image-to-embedding vector model."""
    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)

    net = inputs
    # mean = tf.reduce_mean(net, [1, 2], True)
    # std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    # net = (net - mean) / (std + 1e-5)

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.conv2d(net, 32, [3, 3], scope='conv1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            net = slim.conv2d(net, 128, [3, 3], scope='conv3')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3
            net = slim.flatten(net, scope='flatten')

            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                emb = slim.fully_connected(net, emb_size, scope='fc1')

    return emb


def svhn_model_from_encoder(inputs,
                            is_training=True,
                            augmentation_function=None,
                            emb_size=128,
                            l2_weight=1e-4,
                            img_shape=None,
                            new_shape=None,
                            image_summary=False,
                            num_blocks=None,
                            dropout_keep_prob=1,
                            batch_norm_decay=0.99):  # pylint: disable=unused-argument
    """Same as svhn model, but adapted to autoencoder"""
    inputs = tf.cast(inputs, tf.float32)

    net = inputs

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            with tf.variable_scope('level_1'):
                net = slim.conv2d(net, 32, [3, 3], scope='conv1')
                net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
                net = slim.conv2d(net, 32, [3, 3], scope='conv1_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            with tf.variable_scope('level_2'):
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            with tf.variable_scope('level_3'):
                net = slim.conv2d(net, 128, [3, 3], scope='conv3')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3
            with tf.variable_scope('level_4'):
                net = slim.flatten(net, scope='flatten')
                with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                    emb = slim.fully_connected(net, emb_size, scope='fc1')

    return emb


def dann_model(inputs,
               is_training=True,
               augmentation_function=None,
               emb_size=2048,
               l2_weight=1e-4,
               img_shape=None,
               new_shape=None,
               image_summary=False,
               batch_norm_decay=0.99):  # pylint: disable=unused-argument
    """Construct the image-to-embedding vector model."""
    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)

    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            # TODO(tfrerix) ab hier
            net = slim.conv2d(net, 32, [3, 3], scope='conv1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            net = slim.conv2d(net, 128, [3, 3], scope='conv3')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3
            net = slim.flatten(net, scope='flatten')

            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                emb = slim.fully_connected(net, emb_size, scope='fc1')

    return emb


def stl10_model(inputs,
                is_training=True,
                augmentation_function=None,
                emb_size=128,
                img_shape=None,
                new_shape=None,
                dropout_keep_prob=None,
                num_blocks=None,
                image_summary=False,
                batch_norm_decay=0.99):
    """Construct the image-to-embedding model."""
    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    if is_training and augmentation_function is not None:
        inputs = augmentation_function(inputs, shape)
    if image_summary:
        tf.summary.image('Inputs', inputs, max_outputs=3)
    net = inputs
    net = (net - 128.0) / 128.0
    with slim.arg_scope([slim.dropout], is_training=is_training):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': is_training,
                    'decay': batch_norm_decay
                    },
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(5e-3), ):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.dropout], is_training=is_training):
                    net = slim.conv2d(net, 32, [3, 3], scope='conv_s2')  #
                    net = slim.conv2d(net, 64, [3, 3], stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  #
                    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')  #
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')  #
                    net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                    net = slim.flatten(net, scope='flatten')

                    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                        emb = slim.fully_connected(
                                net, emb_size, activation_fn=None, scope='fc1')
    return emb


def stl10_model_direct(inputs,
                       is_training=True,
                       augmentation_function=None,
                       emb_size=128,
                       img_shape=None,
                       new_shape=None,
                       dropout_keep_prob=None,
                       num_blocks=None,
                       image_summary=False,
                       batch_norm_decay=0.99):
    """Construct the image-to-embedding model."""
    inputs = tf.cast(inputs, tf.float32)

    net = inputs

    with slim.arg_scope([slim.dropout], is_training=is_training):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': is_training,
                    'decay': batch_norm_decay
                    },
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(5e-3), ):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                # 96
                with tf.variable_scope('level_1'):
                    net = slim.conv2d(net, 32, [3, 3], scope='conv_s2')  #
                    net = slim.conv2d(net, 64, [3, 3], stride=2, scope='conv1')
                    # 48
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  #
                    # net = slim.dropout(net, dropout_keep_prob,scope='dropout1')

                # 24
                with tf.variable_scope('level_2'):
                    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')  #

                    # net = slim.dropout(net, dropout_keep_prob, scope='dropout2')
                with tf.variable_scope('level_3'):
                    # 11
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')  #
                    # 5
                    net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                with tf.variable_scope('level_4'):
                    net = slim.flatten(net, scope='flatten')
                    # net = slim.dropout(net, dropout_keep_prob,  scope='dropout3')

                    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                        emb = slim.fully_connected(
                                net, emb_size, activation_fn=None, scope='fc1')
    return emb


def mnist_model(inputs,
                is_training=True,
                emb_size=128,
                l2_weight=1e-3,
                batch_norm_decay=None,
                img_shape=None,
                new_shape=None,
                dropout_keep_prob=None,
                num_blocks=None,
                augmentation_function=None,
                image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32)  # / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    if is_training and augmentation_function is not None:
        tf.map_fn(lambda frame: augmentation_function(frame), inputs)

    if augmentation_function is not None:
        tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs)

    net = inputs
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14

        net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7

        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

        net = slim.flatten(net, scope='flatten')

        emb = slim.fully_connected(net, emb_size, scope='fc1')
    return emb


def mnist_model_dropout(inputs,
                        is_training=True,
                        emb_size=128,
                        l2_weight=1e-3,
                        batch_norm_decay=None,
                        img_shape=None,
                        new_shape=None,
                        dropout_keep_prob=0.8,
                        num_blocks=None,
                        augmentation_function=None,
                        image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32)  # / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    net = inputs

    if is_training and augmentation_function is not None:
        tf.map_fn(lambda frame: augmentation_function(frame), inputs)

    if augmentation_function is not None:
        tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs)

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout2')

            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

            net = slim.flatten(net, scope='flatten')

            # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
            #               scope='dropout3')

            emb = slim.fully_connected(net, emb_size, scope='fc1')

    return emb


def mnist_model_dropout_autoencoder(inputs,
                                    is_training=True,
                                    emb_size=128,
                                    l2_weight=1e-3,
                                    batch_norm_decay=None,
                                    img_shape=None,
                                    new_shape=None,
                                    dropout_keep_prob=0.8,
                                    num_blocks=None,
                                    augmentation_function=None,
                                    image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32)  # / 255.0

    net = inputs

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout2')

            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

            net = slim.flatten(net, scope='flatten')
            # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
            #              scope='dropout3')

            emb = slim.fully_connected(net, emb_size, scope='fc1')

            # DECODE
            dec_emb = slim.fully_connected(emb, 3 * 3 * 128, scope='dec_fc1')
            net = tf.reshape(dec_emb, [-1, 3, 3, 128])

            upsample1 = tf.image.resize_images(net, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Now 7x7x8
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # Now 7x7x8
            upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Now 14x14x8
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # Now 14x14x8
            upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Now 28x28x8
            conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # Now 28x28x16

            logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)

            # decoder

    return logits, emb


def svhn_model_dropout_autoencoder(inputs,
                                   is_training=True,
                                   emb_size=128,
                                   l2_weight=1e-3,
                                   batch_norm_decay=None,
                                   img_shape=None,
                                   new_shape=None,
                                   dropout_keep_prob=0.8,
                                   num_blocks=None,
                                   augmentation_function=None,
                                   image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32)  # / 255.0

    net = inputs

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout2')

            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

            net = slim.flatten(net, scope='flatten')
            # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
            #              scope='dropout3')

            emb = slim.fully_connected(net, emb_size, scope='fc1')

            # DECODE
            dec_emb = slim.fully_connected(emb, 3 * 3 * 128, scope='dec_fc1')
            net = tf.reshape(dec_emb, [-1, 3, 3, 128])

            upsample1 = tf.image.resize_images(net, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Now 7x7x8
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # Now 7x7x8
            upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Now 14x14x8
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # Now 14x14x8
            upsample3 = tf.image.resize_images(conv5, size=(32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Now 28x28x8
            conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            # Now 28x28x16

            logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3, 3), padding='same', activation=None)

            # decoder

    return logits, emb


def mnist_model_dropout_normalized(inputs,
                                   is_training=True,
                                   emb_size=128,
                                   l2_weight=1e-3,
                                   batch_norm_decay=None,
                                   img_shape=None,
                                   new_shape=None,
                                   dropout_keep_prob=0.8,
                                   num_blocks=None,
                                   augmentation_function=None,
                                   image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    net = inputs

    if is_training and augmentation_function is not None:
        tf.map_fn(lambda frame: augmentation_function(frame), inputs)

    if augmentation_function is not None:
        tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs)

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout2')

            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

            net = slim.flatten(net, scope='flatten')

            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout3')

            emb = slim.fully_connected(net, emb_size, scope='fc1')

    return emb


def inception_model(inputs,
                    emb_size=128,
                    is_training=True,
                    end_point='Mixed_7c',
                    augmentation_function=None,
                    img_shape=None,
                    new_shape=None,
                    batch_norm_decay=None,
                    dropout_keep_prob=0.8,
                    num_blocks=None,
                    min_depth=16,
                    depth_multiplier=1.0,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='InceptionV3',
                    num_classes=10,
                    **kwargs):
    from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
    from tensorflow.python.ops import variable_scope
    from tensorflow.contrib.framework.python.ops import arg_scope
    from tensorflow.contrib.layers.python.layers import layers as layers_lib

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)

    inputs = net

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    with variable_scope.variable_scope(
            scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope(
                [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
            _, end_points = inception_v3_base(
                    inputs,
                    scope=scope,
                    min_depth=min_depth,
                    depth_multiplier=depth_multiplier,
                    final_endpoint=end_point)

    net = end_points[end_point]
    net = slim.flatten(net, scope='flatten')
    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc')
    return emb


def inception_model_small(inputs,
                          emb_size=128,
                          is_training=True,
                          **kwargs):
    return inception_model(inputs=inputs, emb_size=emb_size, is_training=is_training,
                           end_point='Mixed_5d', **kwargs)


def vgg16_model(inputs, emb_size=128, is_training=True, img_shape=None, new_shape=None, dropout_keep_prob=0.5,
                num_blocks=None,
                l2_weight=0.0005,
                end_point=None, **kwargs):
    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
                inputs,
                tf.constant(new_shape[:2]),
                method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv1')  # 100
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 50
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 25
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 12
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 6
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 3
            net = slim.flatten(net, scope='flatten')

            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                net = slim.fully_connected(net, 4096, [7, 7], activation_fn=tf.nn.relu, scope='fc6')
                if end_point == 'fc6':
                    return net
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                emb = slim.fully_connected(net, emb_size, [1, 1], activation_fn=None, scope='fc7')

    return emb


def vgg16_model_small(inputs, emb_size=128, is_training=True, img_shape=None, new_shape=None, dropout_keep_prob=0.5,
                      **kwargs):
    return vgg16_model(inputs, emb_size, is_training, img_shape, new_shape, dropout_keep_prob, end_point='fc6',
                       **kwargs)


def alexnet_model(inputs,
                  is_training=True,
                  augmentation_function=None,
                  emb_size=128,
                  l2_weight=1e-4,
                  img_shape=None,
                  new_shape=None,
                  image_summary=False,
                  batch_norm_decay=0.99):
    """Mostly identical to slim.nets.alexnt, except for the reverted fc layers"""

    from tensorflow.contrib import layers
    from tensorflow.contrib.framework.python.ops import arg_scope
    from tensorflow.contrib.layers.python.layers import layers as layers_lib
    from tensorflow.contrib.layers.python.layers import regularizers
    from tensorflow.python.ops import init_ops
    from tensorflow.python.ops import nn_ops
    from tensorflow.python.ops import variable_scope

    trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

    def alexnet_v2_arg_scope(weight_decay=0.0005):
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected],
                activation_fn=nn_ops.relu,
                biases_initializer=init_ops.constant_initializer(0.1),
                weights_regularizer=regularizers.l2_regularizer(weight_decay)):
            with arg_scope([layers.conv2d], padding='SAME'):
                with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def alexnet_v2(inputs,
                   is_training=True,
                   emb_size=4096,
                   dropout_keep_prob=0.5,
                   scope='alexnet_v2'):

        inputs = tf.cast(inputs, tf.float32)
        if new_shape is not None:
            shape = new_shape
            inputs = tf.image.resize_images(
                    inputs,
                    tf.constant(new_shape[:2]),
                    method=tf.image.ResizeMethod.BILINEAR)
        else:
            shape = img_shape
        if is_training and augmentation_function is not None:
            inputs = augmentation_function(inputs, shape)
        if image_summary:
            tf.summary.image('Inputs', inputs, max_outputs=3)

        net = inputs
        mean = tf.reduce_mean(net, [1, 2], True)
        std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
        net = (net - mean) / (std + 1e-5)
        inputs = net

        with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope(
                    [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                    outputs_collections=[end_points_collection]):
                net = layers.conv2d(
                        inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
                net = layers.conv2d(net, 192, [5, 5], scope='conv2')
                net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
                net = layers.conv2d(net, 384, [3, 3], scope='conv3')
                net = layers.conv2d(net, 384, [3, 3], scope='conv4')
                net = layers.conv2d(net, 256, [3, 3], scope='conv5')
                net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')

                net = slim.flatten(net, scope='flatten')

                # Use conv2d instead of fully_connected layers.
                with arg_scope(
                        [slim.fully_connected],
                        weights_initializer=trunc_normal(0.005),
                        biases_initializer=init_ops.constant_initializer(0.1)):
                    net = layers.fully_connected(net, 4096, scope='fc6')
                    net = layers_lib.dropout(
                            net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                    net = layers.fully_connected(net, emb_size, scope='fc7')

        return net

    with slim.arg_scope(alexnet_v2_arg_scope()):
        return alexnet_v2(inputs, is_training, emb_size)
