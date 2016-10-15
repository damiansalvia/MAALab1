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
from Dataset import Dataset

import numpy as np
import csv
from _collections import defaultdict

PARAMETERS = [                                                                                                                                                         
    {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.9, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.001,  'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'constant'  , 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(100,), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,) , 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
#     {'hidden_layer_sizes':(10,8), 'learning_rate':'invscaling', 'learning_rate_init':0.01,   'momentum':0.5, 'random_state':1, 'activation':'logistic', 'solver':'sgd'},
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
    N = 100
    results_random = {"wins":[],"draw":[],"lose":[]}
    results_greedy = {"wins":[],"draw":[],"lose":[]}
    with open('TestCases.csv','wb') as f:
        w = csv.writer(f)
        w.writerow(['Wins','Draw','Lose','Wins','Draw','Lose'])
        for i, parms in enumerate(PARAMETERS):
            name = "Case-%i"%(i+1)
            print name
            
            # Create evaluator (a JugadorGrupo1 player with this parameters for the MLP)
            black_player = EvaluatorClass(name, parms)
            
            # Execute gameplay versus the RandomPlayer
            white_player = RandomPlayer
            gambles = np.array([BatchGame(black_player=black_player, white_player=white_player).play() for _ in xrange(N)])
            wins = 100.0 * np.sum(gambles == GameStatus.BLACK_WINS.value) / N
            draw = 100.0 * np.sum(gambles == GameStatus.DRAW.value      ) / N 
            lose = 100.0 * np.sum(gambles == GameStatus.WHITE_WINS.value) / N
            print "%s vs %s" % (black_player.name.upper(), white_player.name.upper())
            print "Wins:%5.2f%%, Draw:%5.2f%%, Lose:%5.2f%%\n" % (wins,draw,lose)
            results_random['wins'].append(wins)
            results_random['draw'].append(draw)
            results_random['lose'].append(lose)
            
            # Execute gameplay versus the GreedyPlayer
            white_player = GreedyPlayer
            gambles = np.array([BatchGame(black_player=black_player, white_player=white_player).play() for _ in xrange(N)])
            wins2 = 100.0 * np.sum(gambles == GameStatus.BLACK_WINS.value) / N
            draw2 = 100.0 * np.sum(gambles == GameStatus.DRAW.value      ) / N 
            lose2 = 100.0 * np.sum(gambles == GameStatus.WHITE_WINS.value) / N
            print "%s vs %s" % (black_player.name.upper(), white_player.name.upper())
            print "Wins:%5.2f%%, Draw:%5.2f%%, Lose:%5.2f%%\n" % (wins2,draw2,lose2)
            results_greedy['wins'].append(wins2)
            results_greedy['draw'].append(draw2)
            results_greedy['lose'].append(lose2)
            
            w.writerow([wins,draw,lose,wins2,draw2,lose2])
            
    plot_3d_barchart(results_random,title="JugadorGrupo1 vs. RandomPlayer")
    plot_3d_barchart(results_greedy,title="JugadorGrupo1 vs. GreedyPlayer")

    print "Done."            
            