from typing import Union, Tuple

import numpy as np


class SamplePool:
    """
    A Sample Pool maintains a collection of episode starting points, sometimes seeded from past episode completions.
    This allows training to learn across the lifecycle of a cellular automata while 1) making consistent progress and
    learnings in the late stages of the automata and 2) avoiding catastrophic forgetting of the early portions of the
    lifecycle
    """

    def __init__(self, *, _parent: "SamplePool" = None, _parent_idx: int = None, **slots):
        self._parent: "SamplePool" = _parent
        self._parent_idx: int = _parent_idx
        # slots are just key-value constructor arguments where names are the keys
        self._slot_names = slots.keys()
        self._size: int = -1
        for k, v in slots.items():
            if self._size == -1:
                self._size = len(v)
            assert not hasattr(v, '__len__') or self._size == len(v)
            setattr(self, k, np.asarray(v))

    # need to quote "SamplePool" b/c it is a forward reference
    def sample(self, n: Union[int, Tuple[int], None]) -> "SamplePool":
        """
        Returns random episode initialization(s) from the pool
        :param n: The number (or shape) of samples to draw.
        :return: sample(s) as a new sample pool
        """
        idx: int = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)  # TODO why another pool?
        return batch

    def commit(self):
        """

        :return:
        """
        # TODO pydoc: this appears to go through and replace everything in the batch with the 'current' results
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
