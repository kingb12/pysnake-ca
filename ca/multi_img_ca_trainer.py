import random
import traceback
from typing import List, Callable, Tuple, Dict

import numpy as np
import tensorflow as tf

from ca.ca_model import make_seed, to_rgba
from ca.ca_sample_pool import SamplePool
from ca.ca_trainer import DEFAULT_LEARNING_RATE, DEFAULT_POOL_SIZE, DEFAULT_LOG_DIRECTORY, DEFAULT_EPOCHS, \
    DEFAULT_BATCH_SIZE, load_emoji_by_code
from ca.figure_utils import generate_pool_figures, visualize_batch, plot_loss
from ca.multi_img_ca_model import MultiImgCAModel, export_model


class MultiImgCATrainer:
    """

    """

    def __init__(self, model: MultiImgCAModel, target_total_padded_size: Tuple[int, int] = (72, 72),
                 learning_rate=DEFAULT_LEARNING_RATE, pool_size=DEFAULT_POOL_SIZE, batch_size: int = DEFAULT_BATCH_SIZE,
                 log_directory: str = DEFAULT_LOG_DIRECTORY, epochs: int = DEFAULT_EPOCHS):
        # TODO docstring, maybe turn all these parameters into options of some kind
        self.training_images: Dict[str, np.ndarray] = {}
        self.epochs: int = epochs
        self.pool_size = pool_size
        self.target_total_padded_size = target_total_padded_size
        # TODO this type is wrong, is it ndArray instead?
        self.loss_log: List[np.ndarray] = []
        self.learning_rate: float = learning_rate
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
        height, width = self.target_total_padded_size
        if model is None:
            # initialize a model (will use defaults for channel_n, fire_rate)
            model = MultiImgCAModel(height, width)
        self.model: MultiImgCAModel = model
        # initialize the board with a single activated cell
        self.seed: np.ndarray = make_seed(height, width, channel_n=self.model.board_channel_n)
        # initialize some random images
        # create a sample pool for training from this board
        images: List[np.ndarray] = [
            self.random_padded_image() for _ in range(self.pool_size)
        ]
        print('Initializing Pool')
        # TODO: here you might be implicitly loading the pool against well-constructed images of a different organism.
        #  This might be an interesting experiment but probably won't work as expected off the bat.
        self.pool = SamplePool(x=np.repeat(self.seed[None, ...], self.pool_size, axis=0), imgs=images)
        print('Pool initialized')
        # check trainer state
        expected_shape = (height, width, self.model.board_channel_n)
        assert self.seed.shape == expected_shape, f"Board Shape: {self.seed.shape}, Expected: {expected_shape}"
        expected_shape = (self.pool_size, height, width, self.model.board_channel_n)
        assert self.pool.x is not None and self.pool.x.shape == expected_shape, f"Pool X Shape: {self.pool.x.shape}, " \
                                                                                f"Expected: {expected_shape}"

    def loss_f(self, x: tf.Tensor, padded_target: tf.Tensor) -> tf.Tensor:
        """
        Returns the mean-squared-error between the image tensor our board generates
        and the padded target image
        :param padded_target: the padded target image
        :param x: the board at this timestep
        :return: mean-squared-error between our image and the padded target
        """
        # while the boards may not be equal (we don't manage or modify our own board),
        # the shapes should never change
        assert x.shape == (self.batch_size,) + self.seed.shape, "x does not have the shape of a board"
        return tf.reduce_mean(tf.square(to_rgba(x) - padded_target), [-2, -3, -1])

    @tf.function
    def train_step(self, x: tf.Tensor, imgs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                x = self.model(x, imgs)
            loss = tf.reduce_mean(self.loss_f(x, imgs))
        # weights is a synonym of variables and we only want to compute gradient w.r.t. the trainable ones (ignore
        # batchnorm, etc.)
        grads = g.gradient(loss, self.model.trainable_variables)
        grads = [g / (tf.norm(g) + 1e-8) if g is not None else g for g in grads]
        self.trainer.apply_gradients(zip(grads, self.model.trainable_variables))
        return x, imgs, loss

    def train(self):
        """
        Trains the CAModel for
        :return:
        """
        for _ in range(self.epochs + 1):
            batch: SamplePool = self.pool.sample(self.batch_size)
            x0: tf.Tensor = batch.x
            img0: tf.Tensor = batch.imgs
            # This is the sample-pooling strategy implementation: after every episode, we replace our highest-loss
            # batch with the initial seed.
            loss_rank = self.loss_f(x0, img0).numpy().argsort()[::-1]
            x0 = x0[loss_rank]
            img0 = img0[loss_rank]
            # TODO: seed with random image
            x0[:1] = self.seed
            img0[:1] = self.random_padded_image()

            x, imgs, loss = self.train_step(x0, img0)

            # Here we replace everything in thr batch with post-training results
            batch.x[:] = x
            batch.imgs[:] = imgs
            batch.commit()

            step_i: int = len(self.loss_log)
            self.loss_log.append(loss.numpy())

            try:
                if step_i % 20 == 0:
                    generate_pool_figures(self.pool, step_i, self.log_directory)
                if step_i % 200 == 0:
                    # clear_output()
                    visualize_batch(x0, x, step_i, self.log_directory)
                    plot_loss(self.loss_log, self.log_directory)
                    export_model(self.model, self.log_directory + 'train_log/%04d' % step_i)
            except BaseException as e:
                traceback.print_exc()
                continue

            print('\r step: %d, log10(loss): %.3f' % (len(self.loss_log), np.log10(loss)), end='')

    def  random_padded_image(self) -> np.ndarray:
        """
        return an image as a tensor for a randomly selected emoji. The image will be pre-padded to have height and
        width of target_total_padded_size
        :return: padded target image
        """
        emoji_codes: List[str] = ['1f368', '1f458', '002a', '1f36b', '1f38f', '1f459', '1f1f8', '1f410',
                                  '1f3d2', '1f3fa', '1f193', '1f39f', '1f372', '1f1fd', '1f430', '1f378',
                                  '1f460', '1f3c1', '1f17e', '1f453', '1f384', '1f445', '1f45a', '1f3c6',
                                  '1f328', '1f250', '1f42f', '1f170', '1f3f3', '1f377', '1f36d', '1f234',
                                  '1f316', '1f349', '1f380', '1f42b', '1f304', '1f0cf', '1f329', '1f3ec']
        target_code: str = random.choice(emoji_codes)
        # cache downloaded images
        if target_code in self.training_images:
            target_img = self.training_images[target_code]
        else:
            # cache miss: download and create ndarray
            target_img: np.ndarray = load_emoji_by_code(target_code)
            self.training_images[target_code] = target_img
        target_padding_height: int = int((self.target_total_padded_size[0] - target_img.shape[0]) / 2)
        target_padding_width: int = int((self.target_total_padded_size[1] - target_img.shape[1]) / 2)
        padding: np.ndarray = np.array([[target_padding_height, target_padding_height],
                                          [target_padding_width, target_padding_width],
                                          [0, 0]], dtype=np.int32)
        return np.pad(target_img, padding)


if __name__ == '__main__':
    ca_model: MultiImgCAModel = MultiImgCAModel()
    # use defaults otherwise
    ca_trainer: MultiImgCATrainer = MultiImgCATrainer(model=ca_model,
                                                      log_directory='/Users/bking/pysnake-ca/logs/')
    ca_trainer.train()
