# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer

from DataTypes import GameStatus, SquareType
from Game import Game
from UtilsGrupo1 import get_vector


class BatchGameGrupo1(Game):
    DIRS = 8

    def __init__(self, filepath, black_player=RandomPlayer, white_player=RandomPlayer):
        self.players = {SquareType.BLACK: black_player(SquareType.BLACK),
                        SquareType.WHITE: white_player(SquareType.WHITE)}
        self.filepath = filepath # NEW: To generate logfile
        super(BatchGameGrupo1, self).__init__()
        self._last_board = None # NEW: To generate logfile
        self._board_list = [] # NEW: To generate logfile

    def play(self):
        self._last_move = None
        self._last_board = None # NEW: To generate logfile
        while self._game_status == GameStatus.PLAYING:
            if self._state.get_possible_moves(self._turn):
                self._last_move = self.players[self._turn].move(deepcopy(self._state), self._last_move)
                self._do_move(self._last_move, self._turn)
                self._last_board = deepcopy(self._state) # NEW: To generate logfile
            else:
                self._last_move = None
                self._last_board = None # NEW: To generate logfile
            self._pass_turn()
        self._log_to_file()
        if self._game_status == GameStatus.BLACK_WINS:
            self.players[SquareType.WHITE].on_defeat(deepcopy(self._state))
            self.players[SquareType.BLACK].on_win(deepcopy(self._state))
        elif self._game_status == GameStatus.WHITE_WINS:
            self.players[SquareType.WHITE].on_win(deepcopy(self._state))
            self.players[SquareType.BLACK].on_defeat(deepcopy(self._state))
        elif self._game_status == GameStatus.DRAW:
            self.players[SquareType.WHITE].on_draw(deepcopy(self._state))
            self.players[SquareType.BLACK].on_draw(deepcopy(self._state))
        else:
            self.players[SquareType.WHITE].on_error(deepcopy(self._state))
            self.players[SquareType.BLACK].on_error(deepcopy(self._state))
        return self._game_status.value
    
    def _log_to_file(self):
        with open(self.filepath, 'a') as df:
            for board, color in self._board_list: # NEW: To generate logfile
                if board:
                    X = get_vector(board,color)
                    if self._game_status.value == color.value: 
                        y = 1.0  
                    elif self._game_status == GameStatus.DRAW: 
                        y = 0.5  
                    else: 
                        y = 0.0
                    Xy = np.append(X, y).reshape(1,-1)
                    np.savetxt(df, Xy, delimiter=',', newline='\n')
                else: # Pass turn
                    pass
                    
    def _pass_turn(self):
        super(BatchGameGrupo1, self)._pass_turn()
        self._board_list.append((self._last_board, self._turn)) # NEW: To generate logfile
