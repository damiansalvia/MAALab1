# -*- coding: utf-8 -*-

"""Este es un ejemplo de cómo se podrían generar datos para entrenar y evaluar,
    se podría generar corpus de entrenamiento tanto con partidas automáticas o
    jugando en forma interactiva"""

from BatchGame import BatchGame
from InteractiveGame import InteractiveGame

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer
from Players.JugadorGrupo1 import JugadorGrupo1, BatchGameGrupo1
from DataTypes import GameStatus

# Se puede ejecutar una partida interactiva 
# InteractiveGame([GreedyPlayer, RandomPlayer,JugadorGrupo1]).play()
# InteractiveGame([GreedyPlayer, RandomPlayer]).play()

# O se pueden correr varios ejemplos de entrenamiento (o para evaluación)
# GreedyVRandom = [BatchGame(black_player=GreedyPlayer, white_player=RandomPlayer).play() for _ in xrange(100)]
# RandomVGreedy = [BatchGame(black_player=RandomPlayer, white_player=GreedyPlayer).play() for _ in xrange(100)]
# RandomVRandom = [BatchGame(black_player=RandomPlayer, white_player=RandomPlayer).play() for _ in xrange(100)]

games = [
    {'n':100,'agent':JugadorGrupo1,'opponent':RandomPlayer },
#     {'n':100,'agent':RandomPlayer ,'opponent':JugadorGrupo1},
#     {'n':100,'agent':JugadorGrupo1,'opponent':GreedyPlayer },
]
for game in games:
    gambles = [BatchGame(black_player=game['agent'], white_player=game['opponent']).play() for _ in xrange(game['n'])]
    print "%s vs %s" % (game['agent'].name.upper(), game['opponent'].name.upper())
    print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (
        100.0 * len([x for x in gambles if x == GameStatus.BLACK_WINS.value]) / game['n'], 
        100.0 * len([x for x in gambles if x == GameStatus.WHITE_WINS.value]) / game['n'],
        100.0 * len([x for x in gambles if x == GameStatus.DRAW.value      ]) / game['n'],
    )