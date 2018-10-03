from __future__ import division
from __future__ import print_function

import bz2
import gzip
import os

import numpy as np
from tools import data_dirs

DATADIR = data_dirs.norb
NUM_LABELS = 6
IMAGE_SHAPE = [108, 108, 1]


def get_data(name):
    """Utility for convenient data loading."""
    if name == 'unlabeled':
        name = 'train'
    images = np.array([])
    labels = np.array([])
    for fold in ['01', '02']:
        with open(getPath(name, fold, 'dat'), encoding='latin1') as f:
            images = np.append(images, parseNORBFile(f))
        with open(getPath(name, fold, 'cat'), encoding='latin1') as f:
            labels = np.append(labels, parseNORBFile(f))
    images = images.reshape([-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]])
    labels = np.repeat(labels, 2)  # left and right image
    labels = labels.reshape([-1])
    assert labels.shape[0] == images.shape[0]
    return images, labels.astype(np.int32)


def verify(which_set, fileidx, filetype):
    """
    Reads and returns a single file as a np array.
    """
    assert which_set in ['train', 'test']
    assert filetype in ['dat', 'cat']
    assert fileidx in ['01', '02']


def getPath(which_set, fileidx, filetype):
    verify(which_set, fileidx, filetype)
    if which_set == 'train':
        instance_list = '46789'
    elif which_set == 'test':
        instance_list = '01235'
    filename = 'norb-5x%sx9x18x6x2x108x108-%s-%s-%s.mat' % \
               (instance_list, which_set + 'ing', fileidx, filetype)
    path = os.path.join(DATADIR, filename)
    assert os.path.exists(path), "Expected file does not exist: %s" % path
    return path


def readNums(file_handle, num_type, count):
    """
    Reads 4 bytes from file, returns it as a 32-bit integer.
    """
    num_bytes = count * np.dtype(num_type).itemsize
    string = file_handle.read(num_bytes)
    return np.fromstring(string, dtype=num_type)


def readHeader(file_handle, debug=False, from_gzip=None):
    """
    :param file_handle: an open file handle.
    :type file_handle: a file or gzip.GzipFile object

    :param from_gzip: bool or None
    :type from_gzip: if None determine the type of file handle.

    :returns: data type, element size, rank, shape, size
    """

    if from_gzip is None:
        from_gzip = isinstance(file_handle,
                               (gzip.GzipFile, bz2.BZ2File))

    key_to_type = {0x1E3D4C51: ('float32', 4),
                   # what is a packed matrix?
                   # 0x1E3D4C52 : ('packed matrix', 0),
                   0x1E3D4C53: ('float64', 8),
                   0x1E3D4C54: ('int32', 4),
                   0x1E3D4C55: ('uint8', 1),
                   0x1E3D4C56: ('int16', 2)}

    type_key = readNums(file_handle, 'int32', 1)[0]
    elem_type, elem_size = key_to_type[type_key]
    if debug:
        print
        "header's type key, type, type size: ", \
        type_key, elem_type, elem_size
    if elem_type == 'packed matrix':
        raise NotImplementedError('packed matrix not supported')

    num_dims = readNums(file_handle, 'int32', 1)[0]
    if debug:
        print
        '# of dimensions, according to header: ', num_dims

    if from_gzip:
        shape = readNums(file_handle,
                         'int32',
                         max(num_dims, 3))[:num_dims]
    else:
        shape = np.fromfile(file_handle,
                            dtype='int32',
                            count=max(num_dims, 3))[:num_dims]

    if debug:
        print
        'Tensor shape, as listed in header:', shape

    return elem_type, elem_size, shape


def parseNORBFile(file_handle, subtensor=None, debug=False):
    """
    Load all or part of file 'f' into a np ndarray
    :param file_handle: file from which to read file can be opended with
      open(), gzip.open() and bz2.BZ2File() @type f: file-like
      object. Can be a gzip open file.

    :param subtensor: If subtensor is not None, it should be like the
      argument to np.ndarray.__getitem__.  The following two
      expressions should return equivalent ndarray objects, but the one
      on the left may be faster and more memory efficient if the
      underlying file f is big.

       read(f, subtensor) <===> read(f)[*subtensor]

      Support for subtensors is currently spotty, so check the code to
      see if your particular type of subtensor is supported.
      """

    elem_type, elem_size, shape = readHeader(file_handle, debug)
    beginning = file_handle.tell()

    num_elems = np.prod(shape)

    result = None
    if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
        assert subtensor is None, \
            "Subtensors on gzip files are not implemented."
        result = readNums(file_handle,
                          elem_type,
                          num_elems * elem_size).reshape(shape)
    elif subtensor is None:
        result = np.fromfile(file_handle,
                             dtype=elem_type,
                             count=num_elems).reshape(shape)
    elif isinstance(subtensor, slice):
        if subtensor.step not in (None, 1):
            raise NotImplementedError('slice with step', subtensor.step)
        if subtensor.start not in (None, 0):
            bytes_per_row = np.prod(shape[1:]) * elem_size
            file_handle.seek(beginning + subtensor.start * bytes_per_row)
        shape[0] = min(shape[0], subtensor.stop) - subtensor.start
        result = np.fromfile(file_handle,
                             dtype=elem_type,
                             count=num_elems).reshape(shape)
    else:
        raise NotImplementedError('subtensor access not written yet:',
                                  subtensor)

    return result


# Dataset specific augmentation parameters.
augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.2
augmentation_params['brightness_max_delta'] = 0.5
augmentation_params['noise_std'] = 0.05
augmentation_params['flip'] = False
augmentation_params['max_rotate_angle'] = 0