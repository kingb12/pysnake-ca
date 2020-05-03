import math

import numpy as np
import torch

# a blue tile in (0, 1), a red one in tile (0, 0), a green tile in (1, 1)
from torch.ca import CAProtoModel

# since pytorch conv2d prefers NCHW to NHWC, need to restructure target image to be NCHW (C = reds, greens, blues)
REDS = [[1, 0], [1, 0]]
GREENS = [[0, 0], [1, 1]]
BLUES = [[0, 1], [1, 0]]
ACTIVATIONS = [[1, 1], [0, 1]]
BATCH_SIZE = 8
# NHW(4) for 3 = r, g, b, a
TARGET_IMG = torch.tensor(
    np.repeat(np.array([REDS, GREENS, BLUES, ACTIVATIONS])[None, ...], BATCH_SIZE, 0), # N(4)HW
    dtype=torch.float32)



def loss_function(x):
    return torch.mean(torch.pow(to_rgba(x) - TARGET_IMG, 2))


def to_rgba(x: torch.Tensor) -> torch.Tensor:
    """
    Isolate only the red, green, blue, and alpha channels of the current board state x
    :param x: board Tensor
    :return:
    """
    return x[:, :4, ...]


def to_alpha(x: torch.Tensor) -> torch.Tensor:
    """
    Isolate only the alpha channel (clipped) of the current board state x

    :param x: board Tensor
    :return: board(ish)-shaped Tensor of clipped alpha channels
    """
    return torch.clamp(x, 0.0, 1.0, out=None)

def to_rgb(x):
  # assume rgb premultiplied by alpha ?
  rgb, a = x[:, :3, ...], to_alpha(x)
  return 1.0 - a + rgb


def train():
    """
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
    trainer = tf.keras.optimizers.Adam(lr_sched)
    """
    ca = CAProtoModel(2, 2, 12)
    loss_log = []
    lr = 2e-3
    seed = np.zeros([16, 2, 2], np.float32)
    trainer = torch.optim.Adam(ca.parameters(), lr=lr)
    seed[2 // 2, 2 // 2, 3:] = 1.0

    def train_step(x: torch.Tensor):
        # decide the number of iterations in this episode of training
        iter_n = math.floor(np.random.uniform(16, 24))
        for i in torch.arange(0, iter_n):
            x = ca.forward(x)
            loss = torch.mean(loss_function(x))
            # try a few things here:
            # 1. To start, compute backward pass during each step (with/without zero'd gradients)
            # 2. Next, compute backward pass only at the end (with/without zero'd gradients)
            trainer.zero_grad()
            loss.backward(retain_graph=True)
            trainer.step()
        return x, loss
    for i in range(500 + 1):
        x0 = torch.tensor(np.repeat(seed[None, ...], BATCH_SIZE, 0), dtype=torch.float32)
        x, loss = train_step(x0)
        step_i = len(loss_log)
        loss_log.append(loss.detach().numpy())
        if step_i % 5 == 0:
            print(loss_log[-1])

    test = ca(torch.tensor(seed[None, ...], dtype=torch.float32))
    print('Reds:', REDS)
    print('CA Reds:', test[0, 0, ...])
    print('Greens:', GREENS)
    print('CA Greens:', test[0, 1, ...])
    print('Blues:', BLUES)
    print('CA Blues:', test[0, 2, ...])
    print('Activations:', BLUES)
    print('CA Activations:', test[0, 3, ...])

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    train()