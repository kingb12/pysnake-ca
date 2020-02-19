import unittest
from typing import List, Tuple
from game.validator import Board, is_valid_board


class ValidatorTest(unittest.TestCase):
    def test_is_valid_board(self):
        for board, expected in self._is_valid_board_cases():
            self.assertEqual(is_valid_board(board), expected, 'for board:\n {0} and expected: {1}'
                             .format('\n'.join([str(row) for row in board]), expected))

    @staticmethod
    def _is_valid_board_cases() -> List[Tuple[Board, bool]]:
        cases: List[Tuple[Board, bool]] = [
            ([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0]], True),
            ([
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0]], False),
            ([
                 [1, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 1, 0]], True),
            ([
                 [1, 0, 0, 0, 1],
                 [0, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 1, 0]], False),
            ([
                 [1, 0, 0, 0, 1],
                 [1, 1, 0, 0, 1],
                 [0, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 1, 0]], False),
            ([
                [0 for j in range(10)] for i in range(6)], False)
        ]
        return cases


if __name__ == '__main__':
    unittest.main()
