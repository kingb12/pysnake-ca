import numpy as np
import torch

# a blue tile in (0, 1), a red one in tile (0, 0), a green tile in (1, 1)
from ca.proto_ca import CAProtoModel

RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]
WHITE = [1, 1, 1]
TARGET_IMG = torch.tensor(np.array([[RED, BLUE], [WHITE, GREEN]]))

def loss_function(x):
    torch.mean(torch.pow(to_rgba(x) - TARGET_IMG, 2))


def to_rgba(x: torch.Tensor) -> torch.Tensor:
    """
    Isolate only the red, green, blue, and alpha channels of the current board state x
    :param x: board Tensor
    :return:
    """
    return x[..., :4]


def to_alpha(x: torch.Tensor) -> torch.Tensor:
    """
    Isolate only the alpha channel (clipped) of the current board state x

    :param x: board Tensor
    :return: board(ish)-shaped Tensor of clipped alpha channels
    """
    return torch.clamp(x, 0.0, 1.0, out=None)

def to_rgb(x):
  # assume rgb premultiplied by alpha ?
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0 - a + rgb


def train():
    ca = CAProtoModel(2, 2, 12)
    loss_log = []
    lr = 2e-3
    """
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
    trainer = tf.keras.optimizers.Adam(lr_sched)
    """

if __name__ == '__main__':
    train()