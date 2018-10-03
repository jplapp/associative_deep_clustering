########################################################################
#
# Functions for downloading the CIFAR-100 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import pickle

import numpy as np

from semisup.tools import data_dirs
from semisup.tools import download

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

########################################################################
# Various constants used to allocate arrays of the correct size.

NUM_LABELS = 10  # todo only valid for cifar 10
IMAGE_SHAPE = [32, 32, 3]

_params = {
    'cifar10': {
        # Number of files for the training-set.
        '_num_files_train': 5,

        # Number of images for each batch-file in the training-set.
        '_images_per_file': 10000,

        # Number of classes.
        'num_classes': 10,

        'dirname': "cifar-10-batches-py/",

        'label_name': b'labels',

        'test_data_name': 'test_batch',
        # Directory where you want to download and save the data-set.
        # Set this before you start calling any of the functions below.
        'data_path': data_dirs.cifar10,

        # URL for the data-set on the internet.
        'data_url': "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        },
    'cifar100': {
        # Number of files for the training-set.
        '_num_files_train': 1,

        # Number of images for each batch-file in the training-set.
        '_images_per_file': 50000,

        # Number of classes.
        'num_classes': 100,

        'dirname': "cifar-100-python/",

        'label_name': b'fine_labels',

        'test_data_name': 'test',
        # Directory where you want to download and save the data-set.
        # Set this before you start calling any of the functions below.
        'data_path': data_dirs.cifar100,

        # URL for the data-set on the internet.
        'data_url': "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        },
    'cifar100coarse': {
        # Number of files for the training-set.
        '_num_files_train': 1,

        # Number of images for each batch-file in the training-set.
        '_images_per_file': 50000,

        # Number of classes.
        'num_classes': 20,

        'dirname': "cifar-100-python/",

        'label_name': b'coarse_labels',

        'test_data_name': 'test',
        # Directory where you want to download and save the data-set.
        # Set this before you start calling any of the functions below.
        'data_path': data_dirs.cifar100,

        # URL for the data-set on the internet.
        'data_url': "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        }
    }


########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(params, filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(params['data_path'], params['dirname'], filename)


def _unpickle(params, filename):
    """
    Unpickle the given file and return the data.

    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(params, filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        try:
            data = pickle.load(file, encoding='bytes')
        except:
            data = pickle.load(file)  # python2

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-100 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float)  # / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(params, filename):
    """
    Load a pickled data-file from the CIFAR-100 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(params, filename)

    print(data.keys())
    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[params['label_name']])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract(dataset):
    """
    Download and extract the CIFAR-100 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """
    params = _params[dataset]

    download.maybe_download_and_extract(url=params['data_url'], download_dir=params['data_path'])


def load_class_names(params):
    """
    Load the names for the classes in the CIFAR-100 data-set.

    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(params, filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data(dataset):
    """
    Load all the training-data for the CIFAR-100 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.

    @:param dataset: in 'cifar10', 'cifar100'
    """

    params = _params[dataset]
    print(params)
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    # Total number of images in the training-set.
    # This is used to pre-allocate arrays for efficiency.
    _num_images_train = params['_num_files_train'] * params['_images_per_file']

    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(params['_num_files_train']):
        # Load the images and class-numbers from the data-file.
        if dataset == 'cifar10':
            fn = "data_batch_" + str(i + 1)
        else:
            fn = "train"
        images_batch, cls_batch = _load_data(params, fn)

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls


def load_test_data(dataset):
    """
    Load all the test-data for the CIFAR-100 data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    params = _params[dataset]

    images, cls = _load_data(params, filename=params['test_data_name'])

    return images, cls


########################################################################

def get_data(name):
    if name == 'train':
        return load_training_data('cifar10')
    if name == 'test':
        return load_test_data('cifar10')


if __name__ == '__main__':
    dataset = 'cifar100'
    maybe_download_and_extract(dataset)
    images, cls = load_training_data(dataset)

    print(np.shape(images), np.shape(cls))

# Dataset specific augmentation parameters.
augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.2
augmentation_params['brightness_max_delta'] = 0.5
augmentation_params['saturation_lower'] = 0.7
augmentation_params['saturation_upper'] = 1.3
augmentation_params['contrast_lower'] = 0.4
augmentation_params['contrast_upper'] = 1.8
augmentation_params['hue_max_delta'] = 0.2
augmentation_params['noise_std'] = 0.05
augmentation_params['flip'] = True
augmentation_params['max_rotate_angle'] = 10
