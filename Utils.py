'''
Created on 8 oct. 2016

@author: Damian
'''
import numpy as np
from sklearn import preprocessing

from DataTypes import SquareType, GameStatus
from Board import Board
from Move import Move
from BatchGame import BatchGame

import os

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer
from Players.RobotPlayer import RobotPlayer

def get_vector(matrix):
    ret = np.array(matrix).flatten()
    return ret   

def dataset(path='./logs'):
    Generate(path)
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
#                     Xval = normalize(Xval)
#                     print Xval
#                     raw_input()
                    yval = eval(status).value # Deja en {0,1,2} en lugar de GameState.<babla>
                    X.append(Xval) 
                    y.append(yval) 
                else: # Case pass, 3 columns
                    continue
    return X,y

scaler = preprocessing.StandardScaler()
def normalize(X):
    X= np.array([x[0] for x in scaler.fit_transform(X.reshape(-1,1))])
    return X

def Generate(path):
    # Delete previous data
    map( os.unlink, (os.path.join(path,f) for f in os.listdir(path)) )
    # Generate new data
    Play(200,player=GreedyPlayer,opponent=RandomPlayer)
    Play(200,player=RandomPlayer,opponent=RandomPlayer)
    Play(200,player=RandomPlayer,opponent=GreedyPlayer)
#         .Play(player=RobotPlayer,opponent=RobotPlayer)
    
def Play(gambles,player=None,opponent=None):
    wins,lose,draw = 0,0,0
    for _ in xrange(gambles):
        result = BatchGame(black_player=player, white_player=opponent).play()
        wins += 1 if result == GameStatus.BLACK_WINS.value else 0
        lose += 1 if result == GameStatus.WHITE_WINS.value else 0
        draw += 1 if result == GameStatus.DRAW.value       else 0
    print "%s vs %s" % (player.name.upper(), opponent.name.upper())
    print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (100.0*wins/gambles, 100.0*lose/gambles, 100.0*draw/gambles)
    
# Test
# Play(100,player=GreedyPlayer,opponent=RandomPlayer)
# Generate('./logs')
# X,y = dataset()
# print X
# print y
