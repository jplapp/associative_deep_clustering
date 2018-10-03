import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def rotate(x, degrees):
    def _rotate(x, degrees):
        rows, cols = x.shape[:2]
        rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
        res = cv2.warpAffine(x, rot_m, (cols, rows))
        res = np.reshape(res, res.shape[:2] + (x.shape[2],))
        return res

    return tf.py_func(_rotate, [x, degrees], [tf.float32], name='rotate')


def apply_augmentation(image, target_shape, params):
    ap = params
    print(image.shape)
    with tf.name_scope('augmentation'):
        shape = tf.shape(image)

        if "horizontal_move" in ap and ap['horizontal_move']:
          dir = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.int32)
          #dir = dir * tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.int32)  # make zero more likely

          print(dir)
          image = tf.contrib.image.translate(image, [ap['horizontal_move'] * tf.cast(dir[0], tf.float32), 0])

        # rotation
        if ap['max_rotate_angle'] > 0:
            angle = tf.random_uniform(
                    [1],
                    minval=-ap['max_rotate_angle'],
                    maxval=ap['max_rotate_angle'],
                    dtype=tf.float32,
                    seed=None,
                    name='random_angle')
            # tf rotate is reaaally slow
            #image = tf.contrib.image.rotate(image, angle * np.pi / 180, interpolation='BILINEAR')
            r = rotate(image, angle)
            image = r[0]

        # cropping
        if ap['max_crop_percentage']:
            crop_percentage = tf.random_uniform(
                    [1],
                    minval=0,
                    maxval=ap['max_crop_percentage'],
                    dtype=tf.float32,
                    seed=None,
                    name='random_crop_percentage')

            crop_shape = 1.0 - crop_percentage
            crop_shape = tf.cast(shape, tf.float32) * crop_shape
            # assert crop_shape.get_shape() == 2, 'crop shape = {}'.format(crop_shape)
            x = tf.cast(crop_shape, tf.int32)
            cropped_h, cropped_w, _ = tf.unstack(x)
            image = tf.random_crop(image, [cropped_h, cropped_w, target_shape[2]])
            image = tf.image.resize_images(image, target_shape[:2], method=ResizeMethod.NEAREST_NEIGHBOR)

        if "flip" in ap and ap['flip']:
            image = tf.image.random_flip_left_right(image)

        if "saturation_lower" in ap:
            image = tf.image.random_saturation(image, lower=ap["saturation_lower"], upper=ap["saturation_upper"])

        if "brightness_max_delta" in ap:
            image = tf.image.random_brightness(image, max_delta=ap["brightness_max_delta"])

        if "contrast_lower" in ap:
            image = tf.image.random_contrast(image, lower=ap["contrast_lower"], upper=ap["contrast_upper"])

        if "hue_max_delta" in ap:
            image = tf.image.random_hue(image, max_delta=ap["hue_max_delta"])

        noise = tf.random_normal(shape=target_shape, mean=0.0, stddev=ap['noise_std'], dtype=tf.float32)
        image = image + noise

        image.set_shape(target_shape)

        image = tf.clip_by_value(image, -1., 1.)
    return image
