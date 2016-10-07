# -*- coding: utf-8 -*-

from Player import Player
import random


class RandomPlayer(Player):
    """Jugador que elige una jugada al azar dentro de las posibles."""
    name = 'Random'

    def __init__(self, color):
        super(RandomPlayer, self).__init__(self.name, color=color)

    def move(self, board, opponent_move):
        possible_moves = board.get_possible_moves(self.color)
        i = random.randint(0, len(possible_moves) - 1)
        return possible_moves[i]

    def on_win(self, board):
        print 'Gané y soy el color:' + self.color.name

    def on_defeat(self, board):
        print 'Perdí y soy el color:' + self.color.name

    def on_draw(self, board):
        print 'Empaté y soy el color:' + self.color.name

    def on_error(self, board):
        raise Exception('Hubo un error.')
