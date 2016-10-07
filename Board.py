# -*- coding: utf-8 -*-

from DataTypes import SquareType
from Move import Move


class Board(object):

    # Represents de possible directions to move on
    directions = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))

    def __init__(self, rows, cols):
        self._boardData = [[SquareType.EMPTY for _ in xrange(rows)] for _ in xrange(cols)]
        self._cols = 0
        self._rows = 0
        if rows > 0 and cols > 0:
            self._rows = rows
            self._cols = cols
        self.set_position(3, 3, SquareType.WHITE)
        self.set_position(3, 4, SquareType.BLACK)
        self.set_position(4, 3, SquareType.BLACK)
        self.set_position(4, 4, SquareType.WHITE)

    def get_row_count(self):
        return self._rows

    def get_column_count(self):
        return self._cols

    def get_position(self, row, col):
        if 0 <= col < self._cols and 0 <= row < self._rows:
            return self._boardData[row][col]
        return 3

    def set_position(self, row, col, color):
        if 0 <= col < self._cols and 0 <= row < self._rows and 0 <= color.value <= 2:
            self._boardData[row][col] = color
            return color
        return 3

    def reset_board(self):
        self._boardData = [[SquareType.EMPTY for _ in xrange(self._cols)] for _ in xrange(self._rows)]

    def in_board(self, row, col):
        return 0 <= row < self.get_row_count() and 0 <= col < self.get_column_count()

    @staticmethod
    def _get_opposite(color):
        if color == SquareType.WHITE:
            return SquareType.BLACK
        elif color == SquareType.BLACK:
            return SquareType.WHITE
        return color

    def is_valid_move(self, move, color):
        if not move:
            return False
        f = move.get_row()
        c = move.get_col()
        if not self.in_board(f, c) or not self.get_position(f, c) == SquareType.EMPTY:
            return False

        for step in Board.directions:
            row = f + step[0]
            col = c + step[1]
            if self.in_board(row, col) and self.get_position(row, col) == self._get_opposite(color):
                row += step[0]
                col += step[1]
                flag = True
                while self.in_board(row, col) and flag:
                    # reach a piece of the same color
                    if self.get_position(row, col) == color:
                        return True
                    # if the square is empty
                    elif self.get_position(row, col) != self._get_opposite(color):
                        flag = False
                    row += step[0]
                    col += step[1]
        return False

    def get_squares_to_mark(self, move, color):
        f = move.get_row()
        c = move.get_col()
        squares_to_mark = [(f, c)]
        for step in Board.directions:
            row = f + step[0]
            col = c + step[1]
            squares_to_mark_tmp = []
            if self.in_board(row, col) and self.get_position(row, col) == self._get_opposite(color):
                squares_to_mark_tmp.append((row, col))
                row += step[0]
                col += step[1]
                flag = True
                # search for a piece of my color
                while self.in_board(row, col) and flag:
                    if self.get_position(row, col) == color:
                        squares_to_mark += squares_to_mark_tmp
                        flag = False
                    elif self.get_position(row, col) != self._get_opposite(color):
                        flag = False
                    else:
                        squares_to_mark_tmp.append((row, col))
                    row += step[0]
                    col += step[1]
        return squares_to_mark

    def get_possible_moves(self, color):
        moves = []
        for i in xrange(8):
            for j in xrange(8):
                move = Move(i, j)
                if self.is_valid_move(move, color):
                    moves.append(move)
        return moves

    def get_as_matrix(self):
        # noinspection PyUnresolvedReferences
        return [[self._boardData[x][y].value for y in xrange(8)] for x in xrange(8)]

