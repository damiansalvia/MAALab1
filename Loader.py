# -*- coding: utf-8 -*-

"""Este es un ejemplo de cómo se podrían generar datos para entrenar y evaluar,
    se podría generar corpus de entrenamiento tanto con partidas automáticas o
    jugando en forma interactiva"""

from BatchGame import BatchGame
from InteractiveGame import InteractiveGame

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer
from Players.JugadorGrupo1 import JugadorGrupo1
from DataTypes import GameStatus

# Se puede ejecutar una partida interactiva 
# InteractiveGame([GreedyPlayer, RandomPlayer,JugadorGrupo1]).play()
# InteractiveGame([GreedyPlayer, RandomPlayer]).play()

# O se pueden correr varios ejemplos de entrenamiento (o para evaluación)
# GreedyVRandom = [BatchGame(black_player=GreedyPlayer, white_player=RandomPlayer).play() for _ in xrange(100)]
# RandomVGreedy = [BatchGame(black_player=RandomPlayer, white_player=GreedyPlayer).play() for _ in xrange(100)]
# RandomVRandom = [BatchGame(black_player=RandomPlayer, white_player=RandomPlayer).play() for _ in xrange(100)]

# JugadorGrupo1VRandom = [BatchGame(black_player=JugadorGrupo1, white_player=RandomPlayer).play() for _ in xrange(10)]
# JugadorGrupo1VGreedy = [BatchGame(black_player=JugadorGrupo1, white_player=GreedyPlayer).play() for _ in xrange(10)]
# print "Wins %i" % len([x for x in JugadorGrupo1VRandom if x == GameStatus.BLACK_WINS.value])
# print "Wins %i" % len([x for x in JugadorGrupo1VGreedy if x == GameStatus.BLACK_WINS.value])

# JugadorGrupo1VGreedy1 = [BatchGame(black_player=JugadorGrupo1, white_player=GreedyPlayer).play() for _ in xrange(10)]
# JugadorGrupo1VGreedy2 = [BatchGame(black_player=GreedyPlayer, white_player=JugadorGrupo1).play() for _ in xrange(10)]
# print "Wins %i" % len([x for x in JugadorGrupo1VGreedy1 if x == GameStatus.BLACK_WINS.value])
# print "Wins %i" % len([x for x in JugadorGrupo1VGreedy2 if x == GameStatus.WHITE_WINS.value])

JugadorGrupo1VGreedy = [BatchGame(black_player=GreedyPlayer, white_player=JugadorGrupo1).play() for _ in xrange(1)]
print "Wins %i" % len([x for x in JugadorGrupo1VGreedy if x == GameStatus.BLACK_WINS.value])