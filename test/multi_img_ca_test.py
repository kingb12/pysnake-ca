import random
import string
import unittest
from tempfile import TemporaryDirectory
from typing import List

import tensorflow as tf

from ca.multi_img_ca_model import MultiImgCAModel, export_model, reload_model, export_js_model, load_js_model


class MultiImgCaTest(unittest.TestCase):
    """
    Tests for core functions and utilities in MultiImgCAModel
    """

    def test_export_reload(self):
        """
        Test whether we can save and restore models safely
        :return:
        """
        ca: MultiImgCAModel = MultiImgCAModel()
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        key: str = f"{temp_dir}/{''.join(random.sample(string.ascii_lowercase, 10))}"
        weights: List[tf.Variable] = ca.weights
        export_model(ca, key)
        ca_reloaded: MultiImgCAModel = reload_model(key)
        self.assertIsNotNone(ca_reloaded)
        # weights should be equivalent after saving and reloading
        for w, new_w in zip(weights, ca_reloaded.weights):
            tf.assert_equal(w, new_w)
        ca_new: MultiImgCAModel = MultiImgCAModel()
        # these should differ from a brand new model
        not_equal: bool = False
        for w, other_w in zip(weights, ca_new.weights):
            try:
                tf.assert_equal(w, other_w)
            except BaseException as e:
                not_equal = True
                self.assertIsNotNone(e)
        self.assertTrue(not_equal, "expected at least one weight to differ from saved and fresh model")

    def test_export_reload_js(self):
        """
        Test whether we can save and restore models safely (with tfjs)
        :return:
        """
        ca: MultiImgCAModel = MultiImgCAModel()
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        key: str = f"{temp_dir}"
        export_js_model(ca, key)
        ca_reloaded: MultiImgCAModel = load_js_model(key)
        self.assertIsNotNone(ca_reloaded)
        # weights should be equivalent after saving and reloading
        for w, new_w in zip(ca.update_model.weights, ca_reloaded.update_model.weights):
            tf.assert_equal(w, new_w)
        for w, new_w in zip(ca.img_model.weights, ca_reloaded.img_model.weights):
            tf.assert_equal(w, new_w)


if __name__ == '__main__':
    unittest.main()
