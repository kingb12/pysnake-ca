"""
The mission of this prototype is to extend the work described in :cite:`mordvintsev2020growing` to growing new
structures from image input. The hope is that separating the structure definition as provided by an image and the
update policy as provided by a CA provides insight into core mechanisms for learning structured growth. Instead of
training a neural CA against a single initial image, we train against a

NOTE: putting this here b/c I can't think of another place: one thing worth investigating is the cells frame rate vs.
the rate of change: the automata can only move one step at a time in adhering to the rules, but what if it was evaluated
 for a dynamic change every two frames? Every N? At some point, computation would be wasteful and no learning would occur
 but maybe with a different frame rate than the lifecycle, better learning can occur
"""
import json
from typing import List, Tuple

import tensorflow as tf
import tensorflowjs as tfjs
from google.protobuf.json_format import MessageToDict
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    Reshape
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import convert_to_constants

from ca.ca_model import get_living_mask, DEFAULT_CHANNEL_N, DEFAULT_CELL_FIRE_RATE
from ca.ca_model_components import construct_board_update_rule_model, construct_cell_perception_kernel


def batch_wise_convolution(inputs: tf.Tensor, filters: tf.Tensor, dtype: tf.DType = tf.float32) -> tf.Tensor:
    """
    Performs a convolution using a different filter per-batch. Concrete example: in this multi-image CA model, we
    are jointly training an image model that produces a 'filter' representation of an image that helps a universal
    update define an update rule for the CA to produce that image. The image is different per-batch-element, so this
    lets us compute a convolution using a per-element filter.

    :param inputs: input tensor of shape (batch_n, h, w, channel_in)
    :param filters: a tensor of shape (batch_n, filter_h, filter_w, channel_in, channel_out)
    :param dtype: data type of the output tensor
    :return: a 2D convolution on each element in the batch using the filter corresponding to that batch element
    """

    def single_conv(tupl):
        x, kernel = tupl
        return tf.nn.conv2d(x, kernel, strides=(1, 1, 1, 1), padding='VALID')

    return tf.squeeze(tf.map_fn(single_conv, (tf.expand_dims(inputs, 1), filters), dtype=dtype), axis=1)


class MultiImgCAModel(tf.keras.Model):
    """

    """

    def __init__(self, height: int = 3, width: int = 3, board_channel_n: int = DEFAULT_CHANNEL_N,
                 fire_rate: float = DEFAULT_CELL_FIRE_RATE, img_channel_n=72, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.height = height
        self.width = width
        self.board_channel_n: int = board_channel_n
        # separately track just the number of hidden cell channels: in some cases we want to calculate things without
        # interfering with the output cells
        self.board_hidden_channel_n: int = board_channel_n - 4
        self.img_channel_n: int = img_channel_n
        # after image convolution, we should have image channels + (r, g, b) + activation from board
        self.post_img_channel_n: int = img_channel_n + 4
        self.fire_rate = fire_rate
        # Not making this one configurable: without this the update model violates the premise of a cellular automata
        # and can guide dead cells to spawn valid parts of the expected form
        self.apply_living_mask: bool = True

        # img model: model we'll use to process the input img
        self.img_model = self._construct_img_model(self.board_hidden_channel_n, self.img_channel_n)
        # update model: has global awareness of the board and learns an update rule that can be applied to each cell
        self.update_model = construct_board_update_rule_model(self.board_channel_n)
        # perception filter: non-learned filter used to provide info about neighboring cell states to a target cell
        self.perception_kernel = construct_cell_perception_kernel(self.post_img_channel_n)

        # input here is the perceived state of once cell (1 * HWC). Post-perception, (N) batches are cells
        # dummy call to build the model: the number of channels/filters is inferred from the shape of the first calling
        # argument and constructs the weights for the first time
        # TODO standardize this better - 32 is magic
        self(tf.zeros([1, 3, 3, self.board_channel_n]), tf.zeros([1, 72, 72, 4]))

    @staticmethod
    def _img_model_subunit() -> List[Layer]:
        # basic img processing sub-unit stolen from an image classifier tutorial. We may want some thing else dealing
        # with images so small: the max pooling and dropout lose a fair amount of information in this domain
        return [
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25)
        ]

    def _construct_img_model(self, board_hidden_channel_n: int, img_channel_n: int, img_subunit_count: int = 2) -> \
            tf.keras.Model:
        """
        Constructs an image model for this experiment that will be compatible with a board that has board_channel_n
        channels
        :return: keras.Model image processor
        """

        # begin with image processing sub-unit layers n * (normalize -> conv -> pool -> dropout)
        layers: List[Layer] = []
        for _ in range(img_subunit_count):
            layers += self._img_model_subunit()

        # add reshaping unit
        layers += [
            Flatten(),
            Dense(board_hidden_channel_n * img_channel_n, activation='relu'),
            Dropout(0.20),
            # need to condense down to shape (batch size, channel_n, some_num)
            Reshape(target_shape=(board_hidden_channel_n, img_channel_n))
        ]
        return tf.keras.Sequential(layers=layers)

    @tf.function
    def call(self, x: tf.Tensor, images: tf.Tensor, fire_rate: float = None, **kwargs) -> tf.Tensor:
        """
        For a board defined by x, returns the subsequent board x + 1, by first applying perception
        filter and then following the update rule
        :param **kwargs:
        :param x: board of shape (batch_size, h, w, channel_n)
        :param images: image collection of shape
        :param fire_rate: with what frequency should these updates should actually be applied? (cell-by-cell)
        :return: subsequent board state
        """

        # calculate a living mask from step n
        pre_life_mask: tf.Tensor = get_living_mask(x)

        # process the images into filters, inserting two 'empty' dimensions to allow each element in filters to be
        # 1x1 conv2d filter (1, 1, c_in, c_out) on the board
        filters: tf.Tensor = self.img_model(images)[:, None, None, ...]
        # assert filters.shape == (x.shape[0], 1, 1, self.board_hidden_channel_n, self.img_channel_n)

        # compute a 1x1 batch-wise convolution of our board and our image-based filters: this is how the img model
        # communicates to the update model -- TODO: can we do this with frozen weights from an auto-encoder?
        board_output_only: tf.Tensor = x[..., :4]  # only (r, g, b, alpha)
        board_hidden_only: tf.Tensor = x[..., 4:]
        # the image aware board is a batch-wise convolution of 1) our image representation & 2) our current hidden state
        img_aware_board: tf.Tensor = tf.concat([
            board_output_only,
            batch_wise_convolution(board_hidden_only, filters, dtype=x.dtype)
        ], axis=-1)
        perceived: tf.Tensor = self.perceive(img_aware_board)

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
        tf.debugging.assert_equal(x_shape[-1], self.post_img_channel_n)  # verify C, the only known value to the model
        y = tf.nn.depthwise_conv2d(x, self.perception_kernel, [1, 1, 1, 1], 'SAME')
        tf.debugging.assert_shapes([(y, (x_shape[0], x_shape[1], x_shape[2], self.post_img_channel_n * 3))])
        return y


def export_model(ca: MultiImgCAModel, base_filename: str) -> None:
    """
    Saves the weights of the model to a file beginning with base filename (this can be a path). Included
    will be a JSON file `base_filename.json` with the model format, topology, and weights manifest.
    :param ca: The model to export
    :param base_filename: Base file name/path at which to export it
    :return: None
    """
    ca.save_weights(base_filename)

    # TODO: do any of these need adjusting?
    ca.img_model.save_weights(base_filename + '_img')
    ca.update_model.save_weights(base_filename + '_upd')
    ca.save_weights(base_filename + '_all')
    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, ca.board_channel_n]),
        images=tf.TensorSpec([None, None, None, 4]),
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


def reload_model(base_filename: str) -> MultiImgCAModel:
    """
    Reload a model with weights saved at the base file path
    :param base_filename:
    :return:
    """
    ca: MultiImgCAModel = MultiImgCAModel()
    ca.load_weights(base_filename)
    return ca


def export_js_model(ca: MultiImgCAModel, save_dir: str) -> None:
    """
    Export a MultiImgCAModel in a JS readable format

    :param save_dir: where to save the model
    :param ca: model to export
    :return: None
    """
    tfjs.converters.save_keras_model(ca.update_model, f"{save_dir}/update_model")
    tfjs.converters.save_keras_model(ca.img_model, f"{save_dir}/img_model")


def load_js_model(save_dir: str) -> MultiImgCAModel:
    """
    Import a JS readable MultiImgCAModel
    :return: loaded MultiImgCAModel
    """
    # TODO: don't know if this generalizes to unexpected sizes
    ca: MultiImgCAModel = MultiImgCAModel()
    ca.update_model = tfjs.converters.load_keras_model(f"{save_dir}/update_model/model.json",
                                                       use_unique_name_scope=False)
    ca.img_model = tfjs.converters.load_keras_model(f"{save_dir}/img_model/model.json", use_unique_name_scope=False)
    return ca


if __name__ == '__main__':
    model = load_js_model('/Users/bking/Downloads')
