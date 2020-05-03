import unittest

import numpy as np
import torch

from torch.ca import CAProtoModel


class ProtoCaTest(unittest.TestCase):
    def test_perception(self):
        hidden_channel_n = 12
        channel_n = hidden_channel_n + 4
        height, width = 2, 2
        model = CAProtoModel(height, width, hidden_channel_n)
        x = torch.tensor(np.ones((1, channel_n, height, width)), dtype=torch.float32)
        y = model.perceive(x)
        # first filter is identify, should have no impact on ones
        self.assertTrue(torch.allclose(y[0, 0, :, :], x[0, 0, :, :]))
        # verify channel repetition cycles across each filter
        for i in range(3, 3 * channel_n, 3):  # 3, 6...
            self.assertTrue(torch.allclose(y[0, 0, :, :], y[0, i, :, :]))
            self.assertTrue(torch.allclose(y[0, 1, :, :], y[0, i + 1, :, :]))
            self.assertTrue(torch.allclose(y[0, 2, :, :], y[0, i + 2, :, :]))

    def test_model(self):
        hidden_channel_n = 12
        channel_n = hidden_channel_n + 4
        height, width = 2, 2
        model = CAProtoModel(height, width, hidden_channel_n)
        x = torch.tensor(np.ones((1, channel_n, height, width)), dtype=torch.float32)
        y = model.perceive(x)
        dx = model.model(y)
        self.assertEqual(dx.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
