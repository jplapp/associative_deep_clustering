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
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io
from tools import data_dirs

DATADIR = data_dirs.svhn
NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]


def get_data(name, max_num=None):
    """Get a split from the dataset.

    Args:
     name: 'train' or 'test'

    Returns:
     images, labels
    """

    if name == 'train':
        data = scipy.io.loadmat(DATADIR + 'train_32x32.mat')
    elif name == 'unlabeled':
        data = scipy.io.loadmat(DATADIR + 'extra_32x32.mat')
    elif name == 'test':
        data = scipy.io.loadmat(DATADIR + 'test_32x32.mat')


    images = np.rollaxis(data['X'], -1)
    labels = data['y'].ravel() % 10

    num_images = len(images)
    if max_num is not None and num_images > max_num:
        rng = np.random.RandomState()
        choice = rng.choice(len(images), max_num, False)
        images = images[choice]
        labels = labels[choice]


    if name == 'unlabeled':
        return images, None
    else:
        return images, labels


# Dataset specific augmentation parameters.
augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.3
augmentation_params['brightness_max_delta'] = 0.5
augmentation_params['saturation_lower'] = 0.7
augmentation_params['saturation_upper'] = 1.3
augmentation_params['contrast_lower'] = 0.4
augmentation_params['contrast_upper'] = 1.8
augmentation_params['hue_max_delta'] = 0.5
augmentation_params['noise_std'] = 0.05
augmentation_params['flip'] = False
augmentation_params['max_rotate_angle'] = 15

