# -*- coding: UTF-8 -*-

from DataTypes import SquareType, GameStatus
from collections import defaultdict
# from copy import deepcopy
from Board import Board
import datetime
import csv

import os
if os.name == 'nt': format = '%H%M%S%f'
else: format = '%s%f' 

class Game(object):

    def __init__(self):
        self._state = Board(8, 8)
        self._turn = SquareType.BLACK
        self._game_status = GameStatus.PLAYING
        self._last_move = None
        self._move_list = []

    def _log_to_file(self):
#         with open('./logs/log_' + datetime.datetime.now().strftime('%s%f') + '.csv', 'w') as f:
        with open('./logs/log_' + datetime.datetime.now().strftime(format) + '.csv', 'w') as f: # TODO: Delete
            w = csv.writer(f)
            for move, color in self._move_list:
                if move:
                    w.writerow([move.get_row(), move.get_col(), color.name, self._game_status])
                else:
                    w.writerow(['pass', color.name, self._game_status])

    def _do_move(self, move, color):
        if self._state.is_valid_move(move, color):
            for square in self._state.get_squares_to_mark(move, color):
                self._state.set_position(square[0], square[1], color)
            self._game_status = self._finished()
        else:
            if color == SquareType.BLACK:
                self._game_status = GameStatus.WHITE_WINS
            else:
                self._game_status = GameStatus.BLACK_WINS

    def _finished(self):
        if self._state.get_possible_moves(SquareType.BLACK) or self._state.get_possible_moves(SquareType.WHITE):
            return GameStatus.PLAYING
        results = defaultdict(int)
        for i in xrange(8):
            for j in xrange(8):
                results[self._state.get_position(i, j)] += 1
        if results[SquareType.BLACK] > results[SquareType.WHITE]:
            return GameStatus.BLACK_WINS
        if results[SquareType.BLACK] < results[SquareType.WHITE]:
            return GameStatus.WHITE_WINS
        if results[SquareType.BLACK] == results[SquareType.WHITE]:
            return GameStatus.DRAW
        return GameStatus.ERROR

    def _pass_turn(self):
        self._move_list.append((self._last_move, self._turn))
        if self._turn == SquareType.BLACK:
            self._turn = SquareType.WHITE
        elif self._turn == SquareType.WHITE:
            self._turn = SquareType.BLACK
