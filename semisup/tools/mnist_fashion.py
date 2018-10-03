"""
Same as mnist.py, but for the fashion-mnist dataset

"""

from __future__ import division
from __future__ import print_function

import gzip

import numpy as np
from tools import data_dirs

DATADIR = data_dirs.fashion_mnist

NUM_LABELS = 10
IMAGE_SHAPE = [28, 28, 1]


def get_data(name):
    """Utility for convenient data loading."""

    if name == 'train' or name == 'unlabeled':
        return extract_images(DATADIR +
                              '/train-images-idx3-ubyte.gz'), extract_labels(
                DATADIR + '/train-labels-idx1-ubyte.gz')
    elif name == 'test':
        return extract_images(DATADIR +
                              '/t10k-images-idx3-ubyte.gz'), extract_labels(
                DATADIR + '/t10k-labels-idx1-ubyte.gz')


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with open(filename, 'r+b') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with open(filename, 'r+b') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels
