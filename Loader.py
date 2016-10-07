# -*- coding: utf-8 -*-

"""Este es un ejemplo de cómo se podrían generar datos para entrenar y evaluar,
    se podría generar corpus de entrenamiento tanto con partidas automáticas o
    jugando en forma interactiva"""

from BatchGame import BatchGame
from InteractiveGame import InteractiveGame

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer

# Se puede ejecutar una partida interactiva
InteractiveGame([GreedyPlayer, RandomPlayer]).play()

# O se pueden correr varios ejemplos de entrenamiento (o para evaluación)
GreedyVRandom = [BatchGame(black_player=GreedyPlayer, white_player=RandomPlayer).play() for _ in xrange(100)]
RandomVGreedy = [BatchGame(black_player=RandomPlayer, white_player=GreedyPlayer).play() for _ in xrange(100)]
RandomVRandom = [BatchGame(black_player=RandomPlayer, white_player=RandomPlayer).play() for _ in xrange(100)]
