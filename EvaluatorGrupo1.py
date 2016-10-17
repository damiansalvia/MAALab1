# -*- coding: utf-8 -*-

from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import StandardScaler

from Players.JugadorGrupo1 import JugadorGrupo1
from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer

from BatchGame import BatchGame
from DataTypes import GameStatus
from UtilsGrupo1 import plot_3d_barchart
from DatasetGrupo1 import Dataset

import numpy as np
import csv
from collections import defaultdict

PARAMETERS = [                                                                                                                                                         
    {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
    {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
]                                                                                                                                                                      


def EvaluatorClass(nombre, MLPparms):
    def iniciar(self, color):
        self.color = color
        self.name = nombre
        clf = MLPRegressor(**MLPparms)
        self.model = Pipeline([('scl',StandardScaler()),('clf',clf)])
        # Get normalized data
        dataset = Dataset()
        dataset.load(generate=False)
        X,y = dataset.data, dataset.target
        # Fit the classifier with the training data
        self.model.fit(X, y)
    newclass = type(nombre, (JugadorGrupo1,),{"__init__": iniciar})
    return newclass         


if __name__ == '__main__':
    N = 200
#     OPPONENTS = [
#         RandomPlayer,
#         GreedyPlayer
#     ]
#     RESULTS = defaultdict(lambda:{"wins":[],"draw":[],"lose":[]})
#     with open('_results.csv','wb') as f:
#         w = csv.writer(f)
#         w.writerow(['Wins','Draw','Lose']*len(OPPONENTS))
#         for i, parms in enumerate(PARAMETERS):
#             name = "Case-%i"%(i+1)
#             print name
#              
#             # Create evaluator (a JugadorGrupo1 player with this parameters for the MLP)
#             evaluator = EvaluatorClass(name, parms)
#              
#             # Execute gameplay versus the opponent
#             row = []
#             for opponent in OPPONENTS:
#                 print "%s vs %s" % (evaluator.name.upper(), opponent.name.upper())
#                 gambles = []
#                 bar_length = 30
#                 for i in range(N):
#                     percent = float(i) / N
#                     hashes = '#' * int(round(percent * bar_length))
#                     spaces = ' ' * (bar_length - len(hashes))
#                     print "\r[{0}] {1}%".format(hashes + spaces, int(round(percent * 100))),
#                     gambles.append(BatchGame(black_player=evaluator, white_player=opponent).play())
#                 gambles = np.array(gambles) 
#                 wins = 100.0 * np.sum(gambles == GameStatus.BLACK_WINS.value) / N
#                 draw = 100.0 * np.sum(gambles == GameStatus.DRAW.value      ) / N 
#                 lose = 100.0 * np.sum(gambles == GameStatus.WHITE_WINS.value) / N
#                 print "\rWins:%5.2f%%, Draw:%5.2f%%, Lose:%5.2f%%\n" % (wins,draw,lose)
#                 RESULTS[opponent.name]['wins'].append(wins)
#                 RESULTS[opponent.name]['draw'].append(draw)
#                 RESULTS[opponent.name]['lose'].append(lose)
#                 row += [wins,draw,lose]
#              
#             w.writerow(row) 
#                  
#     # Plot results
#     total = len(PARAMETERS)
#     xlabels = ["Case %i"%i for i in xrange(total)]
#     for name,result in RESULTS.items():
#         plot_3d_barchart(result,total,xlabels,title="JugadorGrupo1 vs. %s" % name)
        
    # Plot baseline
    games = [
        {'agent':GreedyPlayer, 'opponent':RandomPlayer },
        {'agent':JugadorGrupo1,'opponent':RandomPlayer},
    ]
    BASELINE = {"wins":[],"draw":[],"lose":[]}
    with open('_baseline.csv','wb') as f:
        w = csv.writer(f)
        w.writerow(['Gameplay','Wins','Draw','Lose'])
        for game in games:
            case = "%s vs %s" % (game['agent'].name.upper(), game['opponent'].name.upper())
            print case
            gambles = np.array([BatchGame(black_player=game['agent'], white_player=game['opponent']).play() for _ in xrange(N)], dtype = np.float64)
            wins = 100.0 * np.sum(gambles == GameStatus.BLACK_WINS.value) / N
            draw = 100.0 * np.sum(gambles == GameStatus.DRAW.value      ) / N 
            lose = 100.0 * np.sum(gambles == GameStatus.WHITE_WINS.value) / N
            BASELINE['wins'].append(wins)
            BASELINE['draw'].append(draw)
            BASELINE['lose'].append(lose)
            print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (wins,draw,lose)
            w.writerow([case,wins,draw,lose])
            
    total = len(BASELINE['wins'])
    xlabels = ["GvsR","JG1vsR"]
    plot_3d_barchart(BASELINE,total,xlabels,title="BASELINE ANALYSIS")
    
    print "Evaluation finished."            
            