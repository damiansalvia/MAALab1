# -*- coding: utf-8 -*-

from Player import Player

from copy import deepcopy
from Utils import get_vector, Dataset

from sklearn.externals import joblib
from sklearn.neural_network.multilayer_perceptron import MLPClassifier,\
    MLPRegressor

import os

class JugadorGrupo1(Player):
    """Jugador que elige una jugada dentro de las posibles utilizando un clasificador."""
    name = 'JugadorGrupo1'

    def __init__(self, color):
        super(JugadorGrupo1, self).__init__(self.name, color=color)
        try:
            self.clf = joblib.load('%s.pkl' % self.name)
        except:
            print "Training...",
#             os.chdir('../') # TODO: Warning
            self.clf =  MLPRegressor(
                            solver='sgd', 
                            hidden_layer_sizes=(10,), 
                            random_state=1,
                            early_stopping = True
                        )
            X,y = Dataset(path='./logs')
            self.clf.fit(X, y)
            joblib.dump(self.clf, '%s.pkl' % self.name)
#             os.chdir('./Players') # TODO: Warning
            print "Done."

    def move(self, board, opponent_move):        
        chosen_move,y_current  = None, None
        for move in board.get_possible_moves(self.color):
            # Simular jugada
            next_board = deepcopy(board)
            for square in board.get_squares_to_mark(move, self.color):
                next_board.set_position(square[0], square[1], self.color)
            # Obtener features y predecir valor
            X = get_vector(next_board.get_as_matrix())
            y = self.clf.predict(X)
            # Quedarse con el mejor
            if not y_current or y_current[0] < y[0]:
                chosen_move = move
                y_current = y
        return chosen_move

    def on_win(self, board):
        print 'Gané y soy el color:' + self.color.name

    def on_defeat(self, board):
        print 'Perdí y soy el color:' + self.color.name

    def on_draw(self, board):
        print 'Empaté y soy el color:' + self.color.name

    def on_error(self, board):
        raise Exception('Hubo un error.')

    
# Test
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from DataTypes import SquareType

# import matplotlib as mlp
# import matplotlib.pyplot as plt
jug = JugadorGrupo1(SquareType.BLACK)
clf = jug.clf

X = [[0,0,0,0,0,0,0,2,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1]]
# X = [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]]
# X = np.array(X).T
# scaler = preprocessing.StandardScaler()
# X = scaler.fit_transform(X).reshape(1,-1)
print X
y = clf.predict(X)
print y
            
            