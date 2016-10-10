# -*- coding: utf-8 -*-

from Player import Player

from copy import deepcopy

import numpy as np

import os
from DataTypes import SquareType


class JugadorGrupo1(Player):
    """Jugador que elige una jugada dentro de las posibles utilizando un clasificador."""
    name = 'JugadorGrupo1'

    def __init__(self, color):
        super(JugadorGrupo1, self).__init__(self.name, color=color)
        self.clf = Classifier(self.name)
#         self.clf.re_train()

    def move(self, board, opponent_move):  
        # For each possible move     
        chosen_move, y_current, X_current  = None, None, None
        for move in board.get_possible_moves(self.color):
            # Simulate gameplay
            next_board = deepcopy(board)
            for square in next_board.get_squares_to_mark(move, self.color):
                next_board.set_position(square[0], square[1], self.color)
            # Get feature vector and predict its value
            X = get_vector(next_board.get_as_matrix(),self.color).reshape(1,-1) # OBS: For 1d problem
            X = self.clf.scl.transform(X)
#             print X
            y = self.clf.clf.predict(X)
            # Keep it if it's better than the current move
            if not y_current or y_current[0] < y[0]:
                chosen_move = move
                X_current = X
                y_current = y
        # Auto-train incrementally with current data       
#         self.clf.clf.partial_fit(X_current,None)
        # Return move
        return chosen_move

    def on_win(self, board):
        print 'Gané y soy el color:' + self.color.name

    def on_defeat(self, board):
        print 'Perdí y soy el color:' + self.color.name

    def on_draw(self, board):
        print 'Empaté y soy el color:' + self.color.name

    def on_error(self, board):
        raise Exception('Hubo un error.')



from sklearn.externals import joblib
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer

class Classifier:
    
    def __init__(self,name):
        
        try:
            self.clf = joblib.load('%s-Model.pkl'  % name)
            self.scl = joblib.load('%s-Scaler.pkl' % name)
        except:
            print "Training...",
#             os.chdir('../') # TODO: Warning - Only test purpose
            self.clf =  MLPRegressor(
                            hidden_layer_sizes=(10,), # Default (100, ) 
                            activation='tanh', # Default: 'relu' 
                            solver='sgd', # Default: 'adam' 
                            alpha=0.0001, 
                            batch_size='auto', 
                            learning_rate='constant', 
                            learning_rate_init=0.001, 
                            power_t=0.5, 
                            max_iter=200, 
                            shuffle=True, 
                            random_state=1, # Default: None 
                            tol=0.0001, 
                            verbose=False, 
                            warm_start=True, # Default: False 
                            momentum=0.9, 
                            nesterovs_momentum=True, 
                            early_stopping=False, 
                            validation_fraction=0.1, 
                            beta_1=0.9, 
                            beta_2=0.999, 
                            epsilon=1e-08
                        )
            self.scl = StandardScaler() # Remains the scaler
            
            # Get normalized data
#             gambles=[
#                 {'n':500,'player':GreedyPlayer,'opponent':RandomPlayer},
#                 {'n':200,'player':RandomPlayer,'opponent':RandomPlayer},
#                 {'n':300,'player':RandomPlayer,'opponent':GreedyPlayer}
#             ]
            dataset = Dataset()
            X,y = dataset.data, dataset.target
            X   = self.scl.fit_transform(X) # Data scaled
            
            # Measure the classifier and fit it
#             show_error(self.clf,X,y)
            self.clf.fit(X, y)
#             show_confusion_matrix(self.clf,X,y)
            
            # Dump the classifier
            joblib.dump(self.clf, '%s-Model.pkl'  % name) # TODO: Descomentar
            joblib.dump(self.scl, '%s-Scaler.pkl' % name) # TODO: Descomentar
#             os.chdir('./Players') # TODO: Warning - Only test purpose
            print "Done."
            
#     def predict(self,X):
#         X = self.scl.transform(X) # Scale the new data
#         return self.clf.predict(X)
    
    def re_train(self):
#         gambles=[
#             {'n':200,'player':JugadorGrupo1,'opponent':RandomPlayer},
#             {'n':200,'player':JugadorGrupo1,'opponent':GreedyPlayer}
#         ]
#         dataset = Dataset(gambles)
#         X,y = dataset.data, dataset.target
#         X = self.scl.transform(X)
#         self.clf.partial_fit(X, y)
        pass
        
#     def partial_fit(self,X,y):
#         X = self.scl.transform(X) # Scale the new data
#         return self.clf.partial_fit(X,y)         
        


# Import players, datatypes and bachgame
from Move import Move
from Board import Board
from DataTypes import GameStatus
from BatchGame import BatchGame

class Dataset:
    
    def __init__(self):
#         os.chdir('../') # TODO: Warning - Only test purpose
        self.generate()
        self._dataset = np.loadtxt('dataset.txt', delimiter=",")
        self.data   = self._dataset[:,:-1]
        self.target = self._dataset[:,-1:].ravel() # OBS: For 1d problem
#         os.chdir('./Players') # TODO: Warning - Only test purpose
        
    def generate(self):
        
        # Delete previous data
        path = './logs'
        map( os.unlink, (os.path.join(path,f) for f in os.listdir(path)) )
         
        # Generate new log data
#         print gambles
#         for gamble in gambles:
#             print gamble['n'], gamble['player'].name, gamble['opponent'].name
#             self.do_play(gamble['n'],black_player=gamble['player'],white_player=gamble['opponent'])
        self.do_play(1000,black_player=GreedyPlayer,white_player=RandomPlayer)
        self.do_play(2500,black_player=RandomPlayer,white_player=RandomPlayer)
        self.do_play(1500,black_player=RandomPlayer,white_player=GreedyPlayer)
        
        # Generate new dataset - dataset.txt
        colors = {'WHITE':SquareType.WHITE,'BLACK':SquareType.BLACK,'EMPTY':SquareType.EMPTY}
        with open('dataset.txt','w') as df:
            for filename in os.listdir(path): # For each log file
                lf = open(''.join([path,'/',filename]),'r')
                # Recreate that gamble
                board = Board(8,8) 
                for line in lf.readlines():
                    values = line.replace("\r\n","").split(',')
                    if len(values) == 4:
                        row, col, color, status = values # Unpack
                        move, color = Move(int(row),int(col)), colors[color]
                        # Simulate gameplay
                        for square in board.get_squares_to_mark(move, color):
                            board.set_position(square[0],square[1], color) 
                        # Get data and target
                        X = get_vector(board.get_as_matrix(),color)
                        y = eval(status).value
                        # Make new file line - csv style
                        Xy = np.append(X, y).reshape(1,-1) # OBS: For getting as row
                        np.savetxt(df, Xy, delimiter=',', newline='\n')
                    else: # Case pass, 3 columns
                        continue
                lf.close()
                
    # Define gambles play
    def do_play(self,n,black_player=RandomPlayer,white_player=RandomPlayer):
        gambles = [BatchGame(black_player=black_player, white_player=white_player).play() for _ in xrange(n)]
        print "%s vs %s" % (black_player.name.upper(), white_player.name.upper())
        print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (
            100.0 * len([x for x in gambles if x == GameStatus.BLACK_WINS.value]) / n, 
            100.0 * len([x for x in gambles if x == GameStatus.WHITE_WINS.value]) / n,
            100.0 * len([x for x in gambles if x == GameStatus.DRAW.value]) / n,
        )  
        
    def stats(self):
        print "******* STATICS *******"
        print "Samples : ",self._dataset.shape[0]
        print "Features: ",self._dataset.shape[1]


##########################################################################################################

other = {SquareType.BLACK:SquareType.WHITE, SquareType.WHITE:SquareType.BLACK}
def get_vector(matrix,color):
    # Defines a feature vector from a matrix board 
    ret = np.array(matrix).flatten()
    return ret   

##########################################################################################################

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def show_error(clf,X,y):
    # Shuffle data
    X, y = shuffle(X, y, random_state=13)
    # Make KFold cross-validation with K=10
    skf = StratifiedKFold(n_splits=10)
    mses = []
    for train, test in skf.split(X, y):
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        mses.append(mse)
        # Get desviance
#         test_score = np.zeros((len(y_test),), dtype=np.float64)
#         print X_test
#         print clf.predict(X_test)
#         for i,y_pred in enumerate(clf.predict(X_test)):
#             print y_test
#             print y_pred
#             test_score[i] = clf.loss_(y_test, y_pred)
#         plot_deviance(clf.train_score_,test_score)
    print mses

def show_confusion_matrix(clf,X,y):
    y_pred = clf.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, 
                          classes=['Gana BLANCO','Gana NEGRO','EMPATE'],
                          title='Matriz de confusion')


def plot_deviance(train_score,test_score):
    N = len(train_score)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Desviacion')
    plt.plot(np.arange(N + 1, train_score, 'b-', label='Desvacion entrenamiento'))
    plt.plot(np.arange(N + 1, test_score, 'r-', label='Desviacion verificacion'))
    plt.legend(loc='upper right')
    plt.xlabel('Iteraciones')
    plt.ylabel('Desviacion')
    

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Real')
    plt.xlabel('Predicho')
        
    
# Test
# import matplotlib as mlp
# import matplotlib.pyplot as plt
# jug = JugadorGrupo1(SquareType.BLACK)
# clf = jug.clf
# d = Dataset()
# d.stats()
# c = Classifier('test')
# c.stats(X, y)
# X = np.array([2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,1.,2.,2.,2.,2.,2.,2.,2.,1.,1.,2.,2.,2.,2.,2.,2.,1.,0.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.])
# X = X.reshape(1,-1)
# print X
# print c.predict(X)



            
            