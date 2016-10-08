# -*- coding: utf-8 -*-

from Player import Player

from copy import deepcopy
from Utils import get_vector

class JugadorGrupo1(Player):
    """Jugador que elige una jugada dentro de las posibles utilizando un clasificador."""
    name = 'JugadorGrupo1'

    def __init__(self, color):
        super(JugadorGrupo1, self).__init__(self.name, color=color)
        self.clf = Classifier(self.name)

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
#             if is_best(y,y_current):
#                 chosen_move = move
#                 y_current = y
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
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from DataTypes import SquareType, GameStatus
from Board import Board
from Move import Move
from BatchGame import BatchGame
import os
    
class Classifier:
     
    def __init__(self,name):
        try:
            self.clf = joblib.load('%s.pkl' % name)
        except:
            print "Training...",
            self.clf =  MLPClassifier(
                            solver='sgd', 
                            hidden_layer_sizes=(10,), 
                            random_state=1,
                            early_stopping = True
                        )
            X,y = self.Dataset()
#             for i,_ in enumerate(X):
#                 print "%s with %s" % (str(X[i]),str(y[i]))
            self.clf.fit(X, y)
#             joblib.dump(self.clf, '%s.pkl' % name)
            print "Done."
             
    def Dataset(self,path='../logs'):
#         self.Generate(path)
        colors = {'WHITE':SquareType.WHITE,'BLACK':SquareType.BLACK,'EMPTY':SquareType.EMPTY}
        X,y = [],[]
        for filename in os.listdir(path):
            with open(''.join([path,'/',filename]),'r') as f:
                # Recreate that gamble
                board = Board(8,8) 
                for line in f.readlines():
                    values = line.replace("\r\n","").split(',')
                    if len(values) == 4:
                        row,col,color,status = values
                        row,col,color = int(row),int(col),colors[color]
                        move = Move(row,col)
                        for square in board.get_squares_to_mark(move, color):
                            board.set_position(square[0],square[1], color) 
                        matrix = board.get_as_matrix()
                        Xval = get_vector(matrix)
                        yval = status 
                        X.append(Xval)
                        y.append(yval) 
                    else: # Case pass, 3 columns
                        continue
        return X,y
    
    def Generate(self,path):
        # Delete previous data
        map( os.unlink, (os.path.join(path,f) for f in os.listdir(path)) )
        # Import other players
        from Players.RandomPlayer import RandomPlayer
        from Players.GreedyPlayer import GreedyPlayer
        from Players.RobotPlayer import RobotPlayer
        # Generate new data
        self.Play(1000,player=GreedyPlayer,opponent=RandomPlayer)
#         self.Play(player=RobotPlayer,opponent=RobotPlayer)
        
    def Play(self,gambles,player=None,opponent=None):
        wins,lose,draw = 0,0,0
        for _ in xrange(gambles):
            result = BatchGame(black_player=player, white_player=opponent).play()
            wins += 1 if result == GameStatus.BLACK_WINS else 0
            lose += 1 if result == GameStatus.WHITE_WINS else 0
            draw += 1 if result == GameStatus.DRAW       else 0
        print "%s vs %s" % (player.name.upper(), opponent.name.upper())
        print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (100.0*wins/gambles, 100.0*lose/gambles, 100.0*draw/gambles)
        
    def predict(self,X):
        return self.clf.predict(X)
     
    def statics(self):
        pass
    
# Test
import numpy as np
clf = Classifier('Test')
X = [[0,0,0,0,0,0,0,2,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1]]
# X = [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]]
# X = np.array(X).T
print clf.predict(X)
            
            