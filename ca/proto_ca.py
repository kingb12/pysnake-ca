from typing import Any

import torch
import numpy as np
from torch.nn.modules.module import T_co


class CAProtoModel(torch.nn.Module):
    def __init__(self, height=2, width=2, hidden=12):
        super(CAProtoModel, self).__init__()
        self.alpha_channel = 0
        self.channel_n = 4 + hidden
        self.update_model = torch.nn.Sequential(
            Conv2D
        )
        self.cell_states = torch.tensor(np.random.rand((height, width, self.channel_n)),
                                        torch.float32, requires_grad=True)

        # TODO: type it
        def perceive(self, x):
            """
            Perceive around the board. We will also use a Sobel operator, like Mordvintsev, et al.
            TODO: once complete, check if Sobel operator was needed, or if diagonals can just be equivalently neighbors

            :param self: the model
            :param x: past cell states
            :return:
            """
            # identifies target cell in 3x3 convolution
            identify = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
            # Sobel filter
            dx = torch.tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0, dtype=torch.float32)
            dy = dx.T
            # define the kernel per cell
            per_cell_kernel = torch.stack([identify, dx, dy], -1)[:, :, None, :]
            # repeat it across an added dimmension for the number of channels
            kernel = per_cell_kernel.repeat(1, 1, self.channel_n, 1)
            # TODO moving this pytorch wont be straigtforward
            """
            y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
            return y
            """

        def forward(self, *cells: Any, **kwargs: Any) -> T_co:

            pass


