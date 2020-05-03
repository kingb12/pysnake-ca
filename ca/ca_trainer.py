from __future__ import annotations

import traceback
from typing import List, Callable, Tuple

import numpy as np
import tensorflow as tf

from ca.ca_model import to_rgba, make_seed, CAProtoModel, export_model, load_image
from ca.ca_sample_pool import SamplePool
from ca.figure_utils import generate_pool_figures, visualize_batch, plot_loss

DEFAULT_LEARNING_RATE = 2e-3
DEFAULT_POOL_SIZE = 1024
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOG_DIRECTORY = "logs/"


class CATrainer:
    """
    Class responsible for training the cellular-automata (holding and applying hyper-parameters, etc.)
    """

    def __init__(self, model: CAProtoModel, target_img: tf.Tensor, target_padding: int = 16,
                 learning_rate=DEFAULT_LEARNING_RATE, pool_size=DEFAULT_POOL_SIZE, use_pattern_pool: bool = False,
                 batch_size: int = DEFAULT_BATCH_SIZE, log_directory: str = DEFAULT_LOG_DIRECTORY):
        # TODO docstring, maybe turn all these parameters into options of some kind
        self.pool_size = pool_size
        self.target_padding = target_padding
        self.target_img = target_img
        self.padded_target = self.__padded_target_image()
        # TODO this type is wrong, is it ndArray instead?
        self.loss_log: List[np.ndarray] = []
        self.learning_rate: float = learning_rate
        self.use_pattern_pool: bool = use_pattern_pool
        self.batch_size: int = batch_size
        self.log_directory = log_directory
        # TODO do we always want a schedule? tweaks?
        self.learning_rate_schedule: Callable[[int], float] = \
            tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                # use given learning rate for 2000 steps, then decrease it by 10x
                [2000], [self.learning_rate, self.learning_rate * 0.1])
        # initialize trainer with our schedule
        self.trainer = tf.keras.optimizers.Adam(self.learning_rate_schedule)
        # image is height * width * depth (r, g, b)
        height, width = self.padded_target.shape[:2]
        if model is None:
            # initialize a model (will use defaults for channel_n, fire_rate)
            model = CAProtoModel(height, width)
        self.model = model
        # initialize the board with a single activated cell
        self.seed: np.ndarray = make_seed(height, width, channel_n=self.model.channel_n)
        # create a sample pool for training from this board
        self.pool = SamplePool(x=np.repeat(self.seed[None, ...], self.pool_size, 0))
        # check trainer state
        expected_shape = (height, width, self.model.channel_n)
        assert self.seed.shape == expected_shape, f"Board Shape: {self.seed.shape}, Expected: {expected_shape}"
        expected_shape = (self.pool_size, height, width, self.model.channel_n)

        assert self.pool.x is not None and self.pool.x.shape == expected_shape, f"Pool X Shape: {self.pool.x.shape}, " \
                                                                    f"Expected: {expected_shape}"

    def __padded_target_image(self) -> tf.Tensor:
        """
        Given the target image represented as a tensor (initialized in {@code self.target_img}
        :return:
        """
        return tf.pad(self.target_img,
                      # in the height and width dimensions, add target_padding padding
                      [(self.target_padding, self.target_padding),
                       (self.target_padding, self.target_padding),
                       (0, 0)])

    def loss_f(self, x: tf.Tensor) -> tf.Tensor:
        """
        Returns the mean-squared-error between the image tensor our board generates
        and the padded target image
        :param x: the board at this timestep
        :return: the
        """
        # while the boards may not be equal (we don't manage or modify our own board),
        # the shapes should never change
        # TODO: should we either 1) use self.board in training, or 2) eliminate it as an attribute and only save shape?
        assert x.shape == (self.batch_size,) + self.seed.shape, "x does not have the shape of a board"
        return tf.reduce_mean(tf.square(to_rgba(x) - self.padded_target), [-2, -3, -1])

    @tf.function()
    def train_step(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = self.model(x)
            loss = tf.reduce_mean(self.loss_f(x))
        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.model.weights))
        return x, loss

    def train(self):
        """

        :return:
        """
        for _ in range(8000 + 1):
            batch: SamplePool = None
            if self.use_pattern_pool:
                batch = self.pool.sample(self.batch_size)
                x0: np.Array = batch.x
                # TODO what's going on here? Its also only used in the Persistent and Regenerating experiments
                # it looks like we find the lowest-loss from the last sample. When we go dynamic we need to know if this
                # loss mgmt strategy is still viable: we compute the min loss on each of the 8 (batch size) examples and
                # start with the least-loss example. Will we know of loss can be computed from a single image/state?
                # are we replacing the lowest/highest loss episode with a new one?
                loss_rank = self.loss_f(x0).numpy().argsort()[::-1]
                x0 = x0[loss_rank]
                x0[:1] = self.seed

            else:
                # this looks like we instead just start over
                x0 = np.repeat(self.seed[None, ...], self.batch_size, 0)

            x, loss = self.train_step(x0)

            if self.use_pattern_pool:
                assert batch is not None, "Expected batch pool to exist already"
                batch.x[:] = x
                batch.commit()

            step_i: int = len(self.loss_log)
            self.loss_log.append(loss.numpy())

            try :
                if step_i % 10 == 0:
                    generate_pool_figures(self.pool, step_i, self.log_directory)
                if step_i % 100 == 0:
                    # clear_output()
                    visualize_batch(x0, x, step_i, self.log_directory)
                    plot_loss(self.loss_log, self.log_directory)
                    export_model(self.model, self.log_directory + 'train_log/%04d' % step_i)
            except BaseException as e:
                traceback.print_exc()
                continue

            print('\r step: %d, log10(loss): %.3f' % (len(self.loss_log), np.log10(loss)), end='')


@tf.function
def make_circle_masks(n, h, w):
    """
    This method is used when applying damage to images. TODO: We'll come back to it in the "Regenerating"
    experiment.
    :param n:
    :param h:
    :param w:
    :return:
    """
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = tf.cast(x * x + y * y < 1.0, tf.float32)
    return mask


def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png' % code
    return load_image(url)


if __name__ == '__main__':
    target_img: tf.Tensor = load_emoji("🦎")
    ca_model: CAProtoModel = CAProtoModel()
    # use defaults otherwise
    ca_trainer: CATrainer = CATrainer(model=ca_model, target_img=target_img,
                                      log_directory='/Users/bking/pysnake-ca/logs/',
                                      use_pattern_pool=True)
    ca_trainer.train()
