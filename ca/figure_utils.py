from typing import List

import PIL.Image, PIL.ImageDraw
import numpy as np
import matplotlib.pyplot as plt

from ca.ca_model import to_rgb
from ca.ca_sample_pool import SamplePool


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
        fmt = 'jpeg'
    f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)


def generate_pool_figures(pool: SamplePool, step_i: int, log_dir: str) -> None:
    # Q: What is x and how is it set? I believe this is the pool image seed set on SamplePool(x=...)
    # TODO what are 72, 49 here? We are leaking board size knowledge, we should find a way to variable-ize this
    tiled_pool = tile2d(to_rgb(pool.x))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
    imwrite(log_dir + 'pools/%04d_pool.jpg' % step_i, tiled_pool)


def visualize_batch(x0, x, step_i: int, log_dir: str):
    """
    For each batch, visualize the input/output pairs of this training session. Writes an image to the log directory
    with a horizontal row of initial states, and below it a horizontal row of the resulting states. For example, the
    first element in each first row may be the single-pixel, used in some trainings to prevent catastrophic forgetting.

    :param x0: the row of initial board states for this batch, which will be translated to images in row 1
    :param x: the row of final board states for this batch, which will be translated to images in row 2
    :param step_i: what training step are we on? Used in file-name for reader convenience
    :param log_dir: where to save the image file in (will be in [log_dir]/train_log/[the filename].jpg)
    :return:
    """
    vis0 = np.hstack(to_rgb(x0).numpy())
    vis1 = np.hstack(to_rgb(x).numpy())
    vis = np.vstack([vis0, vis1])
    imwrite(log_dir + 'batches/batches_%04d.jpg' % step_i, vis)


def plot_loss(loss_log: List[np.ndarray], log_dir: str) -> None:
    """
    Plot the current loss
    :param log_dir: Where to save the figure to
    :param loss_log: log of losses
    :return:
    """
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
    plt.savefig(log_dir + 'loss/loss.png')


def tile2d(a, w=None):
    """
    Reshapes the input vector into a nice grid shape

    :param a: The thing were tiling
    :param w: desired width?
    :return:
    """
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a
