# -*- coding: utf-8 -*-


class Player(object):
    # noinspection SpellCheckingInspection
    """Clase abstracta que deben implementar todos los jugadores."""

    # noinspection SpellCheckingInspection
    name = ''  # En los jugadores a implementar es necesario definir un nombre para la clase.

    def __init__(self, name, color):
        # noinspection SpellCheckingInspection
        """
        Al comenzar el juego, el se invoca esta operacion para
        realizar las tareas de inicialización que sean necesarias.
        :param name: Indica el nombre que identifica a tu jugador.
        :param color: Indica si tu jugador juega con las blancas o las negras.
        """
        self.color = color
        self.name = name

    def move(self, board, opponent_move):
        # noinspection SpellCheckingInspection
        """
        :param board: El estado del tablero en el momento que te toca mover.
        :param opponent_move: El último movimiento que realizó el oponente de tu jugador.
        :return: Una jugada válida.
        :type board: Board
        :type opponent_move: Move
        :rtype: Move
        """
        raise NotImplementedError('You should implement this.')

    def on_win(self, board):
        # noinspection SpellCheckingInspection
        """
        Este método es invocado cuando tu jugador gana.
        :param board: El estado del tablero en el momento de finalizar la partida.
        :return:
        """
        raise NotImplementedError('You should implement this.')

    def on_defeat(self, board):
        # noinspection SpellCheckingInspection
        """
        Este método es invocado cuando tu jugador pierde.
        :param board: El estado del tablero en el momento de finalizar la partida.
        """ 
        raise NotImplementedError('You should implement this.')

    def on_draw(self, board):
        # noinspection SpellCheckingInspection
        """
        Este método es invocado cuando tu jugador empata.
        :param board: El estado del tablero en el momento de finalizar la partida.
        """
        raise NotImplementedError('You should implement this.')

    def on_error(self, board):
        # noinspection SpellCheckingInspection
        """
        Este método es invocado cuando hay un error.
        :param board: El estado del tablero en el momento del error.
        """
        raise NotImplementedError('You should implement this.')
