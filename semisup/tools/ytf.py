"""
YTF dataset, as used in JULE paper

"""

from __future__ import division
from __future__ import print_function

import numpy as np
from tools import data_dirs

DATADIR = data_dirs.ytf

NUM_LABELS = 41
IMAGE_SHAPE = [55, 55, 3]


def get_data(name):
    import h5py
    filename = '/ytf.h5'
    f = h5py.File(DATADIR + filename, 'r')

    # List all groups
    print("Keys: %s" % f.keys())
    data_group_key = list(f.keys())[0]
    label_group_key = list(f.keys())[1]

    # Get the data
    data = np.array(list(f[data_group_key]))
    data = np.swapaxes(data, 1, 3)  # we need channels_last
    labels = np.array(list(f[label_group_key]), np.int)
    unique = list(np.unique(labels))
    l_from_zero = np.array([unique.index(l) for l in np.array(labels, np.int)])

    data = data.transpose(0, 2, 1, 3)
    print(data.shape)

    return data, l_from_zero


# Dataset specific augmentation parameters.
augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.2
augmentation_params['brightness_max_delta'] = 0.5
augmentation_params['saturation_lower'] = 0.7
augmentation_params['saturation_upper'] = 1.3
augmentation_params['contrast_lower'] = 0.4
augmentation_params['contrast_upper'] = 1.8
augmentation_params['hue_max_delta'] = 0.1
augmentation_params['noise_std'] = 0.05
augmentation_params['flip'] = True
augmentation_params['max_rotate_angle'] = 10
