"""
The mission of this prototype is to extend the work described in :cite:`mordvintsev2020growing` to grow dynamic outputs.
This has challenges in that re-generation can't map to a static state. This prototype will begin with a uselessly small
version of these dynamics: TODO TBD

NOTE: putting this here b/c I can't think of another place: one thing worth investigating is the cells frame rate vs.
the rate of change: the automata can only move one step at a time in adhering to the rules, but what if it was evaluated
 for a dynamic change every two frames? Every N? At some point, computation would be wasteful and no learning would occur
 but maybe with a different frame rate than the lifecycle, better learning can occur
"""
import io
import json
from typing import Tuple

import PIL.Image
import PIL.ImageDraw
import numpy as np
import requests
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

from ca.ca_model_components import construct_board_update_rule_model, construct_cell_perception_kernel

DEFAULT_CHANNEL_N = 16
DEFAULT_CELL_FIRE_RATE = 0.5
DEFAULT_TARGET_SIZE = 40


# TODO type all this for mypy
def load_image(url, max_size=DEFAULT_TARGET_SIZE) -> np.ndarray:
    """
    Load an image from the url
    :param url:
    :param max_size:
    :return:
    """
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # pre-multiply RGB by Alpha (image dim 4 is 0 -> 1)
    img[..., :3] *= img[..., 3:]
    return img


def to_rgba(x):
    return x[..., :4]


def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    # assume rgb pre-multiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


def get_living_mask(x: tf.Tensor) -> tf.Tensor:
    """
    Return a living mask
    :param x:
    :return:
    """
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1


def make_seed(height: int, width: int, channel_n: int = DEFAULT_CHANNEL_N) -> np.ndarray:
    """
    Return a seed board initialized with zeros and a single activated cell in the center

    :param width: width of the board
    :param height: height of the board
    :param channel_n: number of channels to support in the board
    :param n: number of boards (e.g. batches)
    :return: initialized board(s)
    """
    x = np.zeros([height, width, channel_n], np.float32)
    x[height // 2, width // 2, 3:] = 1.0
    return x


class CAProtoModel(tf.keras.Model):
    """

    """

    def __init__(self, height: int = 3, width: int = 3, channel_n: int = DEFAULT_CHANNEL_N,
                 fire_rate: float = DEFAULT_CELL_FIRE_RATE):
        super().__init__()
        self.height = height
        self.width = width
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        # Not making this one configurable: without this the update model violates the premise of a cellular automata
        # and can guide dead cells to spawn valid parts of the expected form
        self.apply_living_mask: bool = True

        # update model: has global awareness of the board and learns an update rule that can be applied to each cell
        self.update_model = construct_board_update_rule_model(self.channel_n)
        # perception filter: non-learned filter used to provide info about neighboring cell states to a target cell
        self.perception_kernel = construct_cell_perception_kernel(self.channel_n)

        # input here is the perceived state of once cell (1 * HWC). Post-perception, (N) batches are cells
        self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build model: weights initialized based on first-arg sizes

    @tf.function
    def perceive(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applying our sobel-based perception kernel, return a filtered view from each cell (HWC)
        :param x:
        :return:
        """
        # x has shape NHWC where N = number of batches,
        # HW = height/width of the target image (arbitrary w.r.t. the model)
        # C = number of channels = same as model
        x_shape: Tuple[float] = x.shape
        tf.debugging.assert_equal(len(x_shape), 4)
        tf.debugging.assert_equal(x_shape[-1], self.channel_n) # verify C, the only known value to the model
        y = tf.nn.depthwise_conv2d(x, self.perception_kernel, [1, 1, 1, 1], 'SAME')
        tf.debugging.assert_shapes([(y, (x_shape[0], x_shape[1], x_shape[2], self.channel_n * 3))])
        return y

    @tf.function
    def call(self, x: tf.Tensor, fire_rate: float = None, **kwargs) -> tf.Tensor:
        """
        For a board defined by x, returns the subsequent board x + 1, by first applying perception
        filter and then following the update rule
        :param **kwargs:
        :param x: board
        :param fire_rate: with what frequency should these updates should actually be applied? (cell-by-cell)
        :return: subsequent board state
        """
        # calculate a living mask from step n
        pre_life_mask: tf.Tensor = get_living_mask(x)
        perceived: tf.Tensor = self.perceive(x)
        dx: tf.Tensor = self.update_model(perceived)
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)
        # calculate another from step n + 1
        post_life_mask: tf.Tensor = get_living_mask(x)
        # to be alive you must be 'alive' in both
        living_mask: tf.Tensor = pre_life_mask & post_life_mask
        if self.apply_living_mask:
            x = x * tf.cast(living_mask, tf.float32)
        return x


def export_model(ca: CAProtoModel, base_filename: str) -> None:
    """
    Saves the weights of the model to a file beginning with base filename (this can be a path). Included
    will be a JSON file `base_filename.json` with the model format, topology, and weights manifest.
    :param ca: The model to export
    :param base_filename: Base file name/path at which to export it
    :return: None
    """
    ca.save_weights(base_filename)

    # TODO: do any of these need adjusting?
    cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, ca.channel_n]),
      fire_rate=tf.constant(0.5),
      angle=tf.constant(0.0),
      step_size=tf.constant(1.0))
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
      'format': 'graph-model',
      'modelTopology': graph_json,
      'weightsManifest': [],
    }
    with open(base_filename + '.json', 'w') as f:
        json.dump(model_json, f)


if __name__ == '__main__':
    model = CAProtoModel(2, 2)
    model.update_model.summary()
    print("Perception Kernel Shape: ", model.perception_kernel.shape)
