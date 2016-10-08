'''
Created on 7 oct. 2016

@author: Damian
'''

from BatchGame import BatchGame
from Board import Board
from DataTypes import GameStatus, SquareType
import os
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from Utils import get_vector
from Move import Move

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
             
    def Dataset(self,path='./logs'):
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
            draw += 1 if result == GameStatus.DRAW else 0
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


