import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D


def construct_board_update_rule_model(channel_n: int, high_dim: int = 128) -> tf.keras.Model:
    """
    Constructs a convolution based model for a global update rule applied to each cell
    in the board. The parameters in this model are learned. Note the lack of interaction across
    height and width: only perception filter (before this step) gathers this information. This
    can be seen as a dense linear operation over channels, per cell: 1x1 convolutions project up
    to a high dimension, then back down to channel_n
    :param channel_n:
    :return:
    """
    return tf.keras.Sequential([
        # first 1x1 convolutions projects to a higher dimensional space from our perception vector (relu activated)
        Conv2D(high_dim, (1, 1), activation=tf.nn.relu),
        # second 1x1 convolution projects back down to a cell state size and omits activation
        Conv2D(channel_n, (1, 1), activation=None,
               # second convolution to be initialized with zeros simulates do-nothing behavior by default
               kernel_initializer=tf.zeros_initializer())
    ])


def construct_cell_perception_kernel(channel_n: int) -> tf.Tensor:
    """
    Constructs a convolution-based perception kernel that will be applied to the state grid to allow each
    cell state to perceive its neighbors. This module does not have learned parameters. The primary components are:
    - an identifying matrix (labels the target cell)
    - a sobel filter for the cell's x-wise neighbors
    - a sobel filter for the cells y-wise neighbors

    :param channel_n: number of channels in the state-grid (hidden cells + those used for output)
    :return: a perception kernel (nn.Conv2d) with a repeated stack of channel_n filters (iden, sob_x, sob_y ... with
    shape (channel_n * 3, 1, 3, 3) (since each filter is 3 x 3)
    """
    # identifies target cell in 3x3 convolution
    identify: np.ndarray = np.float32([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # Sobel filters
    sob_x = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    sob_y = sob_x.T
    # we are shaping these to replace the weights in an Conv2D module: stack in (3, 3, 3) such that you can
    # index each filter along the last dimension (e.g. stacked[:, :, 0] == identify)
    stacked = tf.stack([identify, sob_x, sob_y], -1)
    tf.debugging.assert_near(stacked[:, :, 0], identify)
    # add a dimension at index 2
    per_cell_kernel = stacked[:, :, None, :]
    tf.debugging.assert_near(per_cell_kernel[:, :, 0, 1], sob_x)
    # finally, repeat this channel_n times in the third dimension. this is our # of in channels
    kernel = tf.repeat(per_cell_kernel, channel_n, 2)
    assert kernel.shape == (3, 3, channel_n, 3)  # (H = Height, W = Width, C = NumChannels, NumFilters)
    return kernel
