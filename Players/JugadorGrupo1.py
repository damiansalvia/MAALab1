# -*- coding: utf-8 -*-

from Player import Player

from copy import deepcopy
from EntrenadorGrupo1 import Classifier
from Utils import get_vector


class JugadorGrupo1(Player):
    """Jugador que elige una jugada dentro de las posibles utilizando un clasificador."""
    name = 'JugadorGrupo1'

    def __init__(self, color):
        super(JugadorGrupo1, self).__init__(self.name, color=color)
        self.clf = Classifier(self.name)

    def move(self, board, opponent_move):
        possible_moves = board.get_possible_moves(self.color)
        chosen_move = self.choose_move(possible_moves,board,opponent_move)
        return chosen_move

    def on_win(self, board):
        print 'Gané y soy el color:' + self.color.name

    def on_defeat(self, board):
        print 'Perdí y soy el color:' + self.color.name

    def on_draw(self, board):
        print 'Empaté y soy el color:' + self.color.name

    def on_error(self, board):
        raise Exception('Hubo un error.')
    
    def choose_move(self,possible_moves,board,opponent_move):
        for move in possible_moves:
            next_board = deepcopy(board)
            for square in board.get_squares_to_mark(move, self.color):
                next_board.set_position(square[0], square[1], self.color)
            X = get_vector(next_board.get_as_matrix())
            y = self.clf.predict(X)
            
            