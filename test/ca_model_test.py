import unittest
from typing import Tuple

import numpy as np
import tensorflow as tf

from ca.ca_model import get_living_mask


class CaModelTest(unittest.TestCase):
    """
    Tests related to the CAProtoModel class
    """

    def test_get_living_mask(self):
        # check that we mask zeros as dead
        # init a fake batch (n=1) set of 3x3 boards with 8 channels each
        x: tf.Tensor = tf.zeros(shape=(1, 3, 3, 8))
        mask: tf.Tensor = get_living_mask(x)
        assert_shape(mask, (1, 3, 3, 1))
        tf.debugging.assert_type(mask, tf.dtypes.bool)
        tf.debugging.assert_equal(mask, False)
        x = x * tf.cast(mask, tf.float32)
        tf.debugging.assert_equal(x, tf.zeros(shape=(1, 3, 3, 8)))

        # check that we mask ones as living
        x = tf.ones(shape=(1, 3, 3, 8))
        mask = get_living_mask(x)
        assert_shape(mask, (1, 3, 3, 1))
        tf.debugging.assert_type(mask, tf.dtypes.bool)
        tf.debugging.assert_equal(mask, True)

        # check update logic
        board: np.ndarray = tf.constant([[1, 2, 3, 4],
                                      [0, 5, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0]], dtype=tf.float32)
        x = tf.repeat(board[None, :, :, None], 8, axis=3)
        assert_shape(x, (1, 4, 4, 8))
        mask = get_living_mask(x)
        expected: tf.Tensor = tf.constant([[True, True, True, True],
                                           [True, True, True, True],
                                           [True, True, True, False],
                                           [False, False, False, False]])[None, :, :, None]
        tf.debugging.assert_equal(mask, expected)
        x = x * tf.cast(mask, tf.float32)
        tf.debugging.assert_equal(x[0, 0, 0, :], tf.constant([1] * 8, dtype=tf.float32))
        tf.debugging.assert_equal(x[0, 0, 1, :], tf.constant([2] * 8, dtype=tf.float32))
        tf.debugging.assert_equal(x[0, 0, 2, :], tf.constant([3] * 8, dtype=tf.float32))
        tf.debugging.assert_equal(x[0, 0, 3, :], tf.constant([4] * 8, dtype=tf.float32))
        tf.debugging.assert_equal(x[0, 1, 0, :], tf.constant([0] * 8, dtype=tf.float32))
        tf.debugging.assert_equal(x[0, 1, 1, :], tf.constant([5] * 8, dtype=tf.float32))


def assert_shape(tensor: tf.Tensor, shape: Tuple) -> None:
    """
    Assert that the given tensor has the expected shape
    :param tensor: tensor to check shape for
    :param shape: expected shape
    :return: None
    :raise AssertionError: if tensors shape doesn't match expected
    """
    assert tensor.shape == shape, f"Expected shape {tensor.shape} to be {shape}"
