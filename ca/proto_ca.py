from typing import Any

import torch
import torch.nn as nn
import numpy as np


class CAProtoModel(nn.Module):

    def __init__(self, height: int = 2, width: int = 2, hidden: int = 12):
        """
        Returns a CA prototype model with some number of hidden channels
        :param height:
        :param width:
        :param hidden:
        """
        super(CAProtoModel, self).__init__()
        self.alpha_channel = 0
        self.channel_n = 4 + hidden
        self.perception_filter = self._construct_perception_kernel(self.channel_n)
        self.model = self._construct_update_model(self.channel_n)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perceive around the board. See :func:`~_construct_perception_kernel` for details

        :param self: the model
        :param x: past cell states
        :return:
        """
        return self.perception_filter(x)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a single update to the past grid state (x) to get the next grid state (x + 1)
        :param x: grid
        :return: updated grid
        """
        dx = self.model(x)
        x += dx
        return x

    @staticmethod
    def _construct_perception_kernel(channel_n: int) -> nn.Module:
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
        identify = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        # Sobel filters
        sob_x = torch.tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0, dtype=torch.float32)
        sob_y = sob_x.T
        # we are shaping these to replace the weights in an nn.Conv2d module: stack in  (3, 3, 3) such that you can
        # index each filter along the first dimmension (e.g. stacked[0, :, :] == identify)
        stacked = torch.stack([identify, sob_x, sob_y], 0)
        assert torch.allclose(stacked[0, :, :], identify)
        # add a dimension at index 1 (this is in_channels / groups in size and thus should remain == 1)
        per_channel_kernel = stacked[:, None, :, :]
        assert torch.allclose(per_channel_kernel[1, 0, :, :], sob_x)
        # finally, repeat this channel_n times in the first dimmension. this is our # of out channels where we produce
        # 3 channels per in channel with filters (identify, sob_x, sob_y) applied to each
        kernel = per_channel_kernel.repeat(channel_n, 1, 1, 1)
        assert torch.allclose(kernel[2, 0, :, :], kernel[5, 0, :, :]) and torch.allclose(kernel[2, 0, :, :], sob_y)
        # replace weights in nn.Conv2d. Remember: no bias and disable gradient calcs
        conv = nn.Conv2d(in_channels=channel_n, out_channels=3 * channel_n, kernel_size=3, groups=16, bias=False,
                         padding=1)
        conv.weight = nn.Parameter(kernel, requires_grad=False)
        return conv

    @staticmethod
    def _construct_update_model(channel_n: int):
        """

        :param channel_n:
        :return:
        """
        # we want the second convolution to be initialized with zeros to begin with do-nothing behavior by default:
        # changes from this
        conv2 = nn.Conv2d(in_channels=128, out_channels=channel_n, kernel_size=1)
        nn.init.zeros_(conv2.weight)

        # using 1 x 1 convolutions instead of dense layers for performance reasons (cited in paper example) # TODO why?
        return nn.Sequential(
            # first 1x1 convolutions projects to a higher dimmensional space from our perception vector
            nn.Conv2d(in_channels=channel_n * 3, out_channels=128, kernel_size=1),
            nn.ReLU(),
            # second 1x1 convolution projects back down to a cell state size and omits activation
            conv2
        )
        # per paper, we could do this with 2 Linear layers from 48 -> 128 -> 16 but its slower. board is then batches

        # TODO how does above update rule not cross-reference data from multiple perception vectors? is this in the
        #  1x1 conv definition?

    def forward(self, *x: Any, **kwargs: Any):
        """
        First, apply perception to each cell state. Then apply update.
        :param x:
        :param kwargs:
        :return:
        """
        x: torch.Tensor
        perceived = self.perceive(x)
        return self.update(perceived)


if __name__ == '__main__':
    x = torch.tensor(np.ones((1, 16, 2, 2)), dtype=torch.float32)
    model = CAProtoModel()
    y = model.perceive(x)
