# -*- coding: utf-8 -*-

from Player import Player

from copy import deepcopy
import numpy as np

from sklearn.externals import joblib
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class JugadorGrupo1(Player):
    """Jugador que elige una jugada dentro de las posibles utilizando un clasificador."""
    name = 'JugadorGrupo1'

    def __init__(self, color):
        super(JugadorGrupo1, self).__init__(self.name, color=color)
        try: # Try to load from file
            self.model = joblib.load('%s.pkl'  % self.name)
        except:
            print "Training...",
#             os.chdir('../') # TODO: Warning - Only test purpose
            clf = MLPRegressor(
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
            self.model = Pipeline([('scl',StandardScaler()),('clf',clf)])
            
            # Get normalized data
            dataset = Dataset()
            X,y     = dataset.data, dataset.target
            
            # Fit the classifier
            self.model.fit(X, y)
            
            # Dump the classifier
            joblib.dump(self.model, '%s.pkl'  % self.name) # TODO: Descomentar
#             os.chdir('./Players') # TODO: Warning - Only test purpose
            print "Done."

    def move(self, board, opponent_move):  
        # For each possible move     
        chosen_move, y_current, X_current  = None, None, None
        for move in board.get_possible_moves(self.color):
            # Simulate gameplay
            next_board = do_move(board,move,self.color)
            # Get feature vector and predict its value
            X = get_vector(next_board,self.color).reshape(1,-1) # OBS: For 1d problem
#             print X
            y = self.model.predict(X)
            # Keep it if it's better than the current move
            if not y_current or y_current[0] < y[0]:
                chosen_move = move
                X_current = X
                y_current = y
        # Auto-train incrementally with current data       
#         self.model.partial_fit(X_current,None)
        # Return move
        return chosen_move

    def on_win(self, board):
        #print 'Gané y soy el color:' + self.color.name
        pass
        
    def on_defeat(self, board):
#         print 'Perdí y soy el color:' + self.color.name
        pass

    def on_draw(self, board):
#         print 'Empaté y soy el color:' + self.color.name
        pass

    def on_error(self, board):
        raise Exception('Hubo un error.')

##########################################################################################################

import os     
from sklearn.metrics.regression import mean_squared_error

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer 

from DataTypes import SquareType, GameStatus

COLORS = {'WHITE':SquareType.WHITE,'BLACK':SquareType.BLACK,'EMPTY':SquareType.EMPTY}

class Dataset:
    
    def __init__(self,generate=True,logfile='_logfile.csv'):
#         os.chdir('../') # TODO: Warning - Only test purpose
        self.filepath = logfile
        dataset = self.load(generate)
        self.data   = dataset[:,:-1]
        self.target = dataset[:,-1:].ravel() # OBS: For 1d problem
#         os.chdir('./Players') # TODO: Warning - Only test purpose
        
    def load(self,generate):        
        if generate:
            # Delete previous data
            if os.path.exists(self.filepath): os.remove(self.filepath)
            # Generate new log data
            self.do_play(200,black_player=GreedyPlayer,white_player=RandomPlayer)
            self.do_play(800,black_player=RandomPlayer,white_player=RandomPlayer)
        return np.loadtxt(self.filepath, delimiter=",")
    
    def do_play(self,n,black_player=RandomPlayer,white_player=RandomPlayer):
        gambles = [BatchGameGrupo1(self.filepath, black_player=black_player, white_player=white_player).play() for _ in xrange(n)]
        print "\r%s vs %s" % (black_player.name.upper(), white_player.name.upper()),
        print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (
            100.0 * len([x for x in gambles if x == GameStatus.BLACK_WINS.value]) / n, 
            100.0 * len([x for x in gambles if x == GameStatus.WHITE_WINS.value]) / n,
            100.0 * len([x for x in gambles if x == GameStatus.DRAW.value      ]) / n,
        )

##########################################################################################################

def do_move(board,move,color):
    next_board = deepcopy(board)
    for square in next_board.get_squares_to_mark(move,color):
        next_board.set_position(square[0], square[1], color)
    return next_board
    
OTHER = {SquareType.BLACK:SquareType.WHITE, SquareType.WHITE:SquareType.BLACK}
WEIGHT_MATRIX = np.array([
        [50,-1, 5, 2, 2, 5,-1,50],
        [-1,-5, 1, 1, 1, 1,-5,-1],
        [ 5, 1, 1, 1, 1, 1, 1, 5],
        [ 2, 1, 1, 1, 1, 1, 1, 2],
        [ 2, 1, 1, 1, 1, 1, 1, 2],
        [ 5, 1, 1, 1, 1, 1, 1, 5],
        [-1,-5, 1, 1, 1, 1,-5,-1],
        [50,-1, 5, 2, 2, 5,-1,50]
    ])
def get_vector(board,color):
    # Get the current board as a matrix and get the influence map sum of both players 
    matrix = np.array(board.get_as_matrix()) 
    Ia = np.sum( (matrix == color.value       ) * WEIGHT_MATRIX )
    Io = np.sum( (matrix == OTHER[color].value) * WEIGHT_MATRIX )
    # Simulate the best greedy move for the opponent
    max_squares = 0
    chosen_move = None
    for move in board.get_possible_moves(OTHER[color]):
        tmp = len(board.get_squares_to_mark(move=move, color=OTHER[color]))
        if max_squares < tmp:
            chosen_move = move 
            max_squares = tmp
    # Get the feature vector
    if max_squares == 0 and not board.get_possible_moves(color):
        # The opponent doesn't move, and the agent neighter (gameplay ends)
        Imax = np.sum(np.abs(WEIGHT_MATRIX))
        if np.sum(matrix == color.value) > np.sum(matrix == OTHER[color].value):
            vector = [Ia,Io,Imax,0]
        else:
            vector = [Ia,Io,0,Imax]
    elif max_squares == 0:
        # The opponent doesn't move, but the agent does (gameplay continues)
        vector = [Ia,Io,Ia,Io]
    else:
        # The opponent moves
        current_board = do_move(board, chosen_move, OTHER[color])
        matrix = np.array(current_board.get_as_matrix()) 
        NIa = np.sum( (matrix == color.value       ) * WEIGHT_MATRIX )
        NIo = np.sum( (matrix == OTHER[color].value) * WEIGHT_MATRIX )
        vector = [Ia,Io,NIa,NIo]
    # Return the vector
    return np.array(vector)   

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
        
##########################################################################################################      

from Game import Game
class BatchGameGrupo1(Game):
    DIRS = 8

    def __init__(self, filepath, black_player=RandomPlayer, white_player=RandomPlayer):
        self.players = {SquareType.BLACK: black_player(SquareType.BLACK),
                        SquareType.WHITE: white_player(SquareType.WHITE)}
        self.filepath = filepath
        super(BatchGameGrupo1, self).__init__()
        self._last_board = None # NEW: To generate logfile
        self._board_list = [] # NEW: To generate logfile

    def play(self):
        self._last_move = None
        self._last_board = None # NEW: To generate logfile
        while self._game_status == GameStatus.PLAYING:
            if self._state.get_possible_moves(self._turn):
                self._last_move = self.players[self._turn].move(deepcopy(self._state), self._last_move)
                self._do_move(self._last_move, self._turn)
                self._last_board = deepcopy(self._state) # NEW: To generate logfile
            else:
                self._last_move = None
                self._last_board = None # NEW: To generate logfile
            self._pass_turn()
        self._log_to_file()
        if self._game_status == GameStatus.BLACK_WINS:
            self.players[SquareType.WHITE].on_defeat(deepcopy(self._state))
            self.players[SquareType.BLACK].on_win(deepcopy(self._state))
        elif self._game_status == GameStatus.WHITE_WINS:
            self.players[SquareType.WHITE].on_win(deepcopy(self._state))
            self.players[SquareType.BLACK].on_defeat(deepcopy(self._state))
        elif self._game_status == GameStatus.DRAW:
            self.players[SquareType.WHITE].on_draw(deepcopy(self._state))
            self.players[SquareType.BLACK].on_draw(deepcopy(self._state))
        else:
            self.players[SquareType.WHITE].on_error(deepcopy(self._state))
            self.players[SquareType.BLACK].on_error(deepcopy(self._state))
        return self._game_status.value
    
    def _log_to_file(self):
        with open(self.filepath, 'a') as df:
            for board, color in self._board_list: # NEW: To generate logfile
                if board:
                    X = get_vector(board,color)
                    if self._game_status.value == color.value: 
                        y = 1.0  
                    elif self._game_status == GameStatus.DRAW: 
                        y = 0.5  
                    else: 
                        y = 0.0
                    Xy = np.append(X, y).reshape(1,-1)
                    np.savetxt(df, Xy, delimiter=',', newline='\n')
                else: # Pass turn
                    pass
                    
    def _pass_turn(self):
        super(BatchGameGrupo1, self)._pass_turn()
        self._board_list.append((self._last_board, self._turn)) # NEW: To generate logfile



 
# Test
# import matplotlib as mlp
# import matplotlib.pyplot as plt
# jug = JugadorGrupo1(SquareType.BLACK)
# clf = jug.clf
# d = Dataset(generate=False,path='../logs')
# print d.target
# print d.stats()
# print np.array([np.append(x,y) for x,y in zip(d.data,d.target)])
# c = Classifier('test')
# c.stats(X, y)
# X = np.array([2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,1.,2.,2.,2.,2.,2.,2.,2.,1.,1.,2.,2.,2.,2.,2.,2.,1.,0.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.])
# X = X.reshape(1,-1)
# print X
# print c.predict(X)



            
            