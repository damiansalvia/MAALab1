# -*- coding:utf-8 -*-

from Player import Player


class GreedyPlayer(Player):
    """Jugador que siempre elige la jugada que más fichas come."""

    name = 'Greedy'

    def __init__(self, color):
        super(GreedyPlayer, self).__init__(self.name, color)

    def move(self, board, opponent_move):
        """
        :param board: Board
        :param opponent_move: Move
        :return: Move
        """
        max_squares = 0
        chosen_move = None
        for move in board.get_possible_moves(self.color):
            tmp = len(board.get_squares_to_mark(move=move, color=self.color))
            if max_squares < tmp:
                chosen_move = move
                max_squares = tmp
        return chosen_move

    def on_win(self, board):
        pass #print 'Gané y soy el color:' + self.color.name

    def on_defeat(self, board):
        pass #print 'Perdí y soy el color:' + self.color.name

    def on_draw(self, board):
        pass #print 'Empaté y soy el color:' + self.color.name

    def on_error(self, board):
        raise Exception('Hubo un error.')
