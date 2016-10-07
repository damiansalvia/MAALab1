# -*- coding: utf-8 -*-

from Player import Player
from sklearn.neighbors import KNeighborsClassifier
import random


class JugadorGrupo1(Player):
    """Jugador que elige una jugada dentro de las posibles utilizando aprendizaje automatico."""
    name = 'JugadorGrupo1'

    def __init__(self, color):
        super(JugadorGrupo1, self).__init__(self.name, color=color)

    def move(self, board, opponent_move):        
        max_squares = 0
        chosen_move = None
        for move in board.get_possible_moves(self.color):
            tmp = len(board.get_squares_to_mark(move=move, color=self.color))
            if max_squares < tmp:
                chosen_move = move
                max_squares = tmp
        return chosen_move

    def on_win(self, board):
        print 'Gané y soy el color:' + self.color.name

    def on_defeat(self, board):
        print 'Perdí y soy el color:' + self.color.name

    def on_draw(self, board):
        print 'Empaté y soy el color:' + self.color.name

    def on_error(self, board):
        raise Exception('Hubo un error.')
