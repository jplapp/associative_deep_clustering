"""

Definitions and utilities for COIL data. We use all data for train and test (just following literature)

"""

from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from tools import data_dirs

DATADIR = data_dirs.coil20

NUM_LABELS = 20
IMAGE_SHAPE = [32, 32, 1]


def get_data(name):
    """Utility for convenient data loading."""

    images, labels = extract_images()
    images = np.reshape(images, list(images.shape) + [1, ])

    rng = np.random.RandomState(seed=47)
    # inds = rng.choice(len(images), int(len(images) / 5 * 4))

    # print(inds)
    if name == 'train' or name == 'unlabeled':
        return images, labels
    elif name == 'test':
        return images, labels


def extract_images():
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""

    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]

    print(DATADIR)
    images = []
    labels = []

    for file in files:
        img = cv2.imread(join(DATADIR, file), 0)
        img = cv2.resize(img, (32, 32))

        images.append(img)

        img_id = file[3:5]
        if img_id[-1] == '_':
            img_id = img_id[0]

        labels.append(int(img_id) - 1)

    return np.array(images), np.array(labels)


# Dataset specific augmentation parameters.
augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.2
augmentation_params['brightness_max_delta'] = 0.5
augmentation_params['noise_std'] = 0.05
augmentation_params['flip'] = True
augmentation_params['max_rotate_angle'] = 10
