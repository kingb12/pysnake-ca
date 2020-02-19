import numpy as np
from typing import List, Tuple, Set, Iterable

Board = List[List[int]]
Cell = Tuple[int, int]
BoardSet = List[List[Cell]]
Snake = Set[Cell]


def is_valid_board(board: Board) -> bool:
    # TODO types
    """

    :param board:
    :return:
    """
    # start with board set. for each cell, check its neighbors. If i
    food: Cell = None
    snake: Snake = None
    rows = len(board)
    cols = len(board[0])
    for i in range(rows):
        # we want to snake through the board
        is_forward_row: bool = i % 2 == 0
        col_range: Iterable = range(0, cols) if is_forward_row else range(cols, 0, -1)
        for j in col_range:
            if board[i][j]:
                # active cell, check if snake or food
                neighbors = get_neighbors(board, (i, j))
                if len(neighbors) == 0:
                    # this is the potential food
                    if food is not None:
                        # can't have two isolated cells
                        return False
                    food = (i, j)
                if snake is not None:
                    # we've already found the whole snake
                    return False
                snake = {(i, j)}
                for n in neighbors:
                    snake.add(n)
                # now do a BFS on neighbors, and add them all
                neighborhood_queue: List[Cell] = [n for n in neighbors if not n in snake]
                while len(neighborhood_queue) > 0:
                    cell = neighborhood_queue.pop(0)
                    neighbors = get_neighbors(board, cell)
                    for n in neighbors:
                        if n not in snake:
                            snake.add(n)
                            neighborhood_queue.append(n)
        return snake is not None


def get_neighbors(board: Board, cell: Cell) -> List[Cell]:
    """
    return the number of valid neighbors at this spot in the snake board

    :param board: snake board
    :param cell: target cell
    :return: number of neighbors of the target cell
    """
    i = cell[0]
    j = cell[1]
    rows = len(board)
    cols = len(board[0])
    # L, R, U, D
    potential_neighbors = [(x, y) for x, y
                           in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                           if 0 <= x < rows and 0 <= y < cols]
    result = []
    for x, y in potential_neighbors:
        if board[x][y]:
            result.append((x, y))
    return result


def new_board_set(board: Board) -> BoardSet:
    # initialize a board
    assert board is not None, 'null board'
    # todo assert type?
    assert len(board) > 0 and len(board[0]) > 0, 'empty or undefined board'
    rows = len(board)
    cols = len(board[0])
    # initialize with their current coordinates
    return [[(i, j) for j in range(cols)] for i in range(rows)]


if __name__ == '__main__':
    board: Board = [[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0]]
    assert is_valid_board(board)
