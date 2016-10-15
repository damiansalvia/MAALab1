# -*- coding: utf-8 -*-

import os     
import numpy as np

from Players.RandomPlayer import RandomPlayer
from Players.GreedyPlayer import GreedyPlayer 

from DataTypes import SquareType, GameStatus
from BatchGameGrupo1 import BatchGameGrupo1

COLORS = {'WHITE':SquareType.WHITE,'BLACK':SquareType.BLACK,'EMPTY':SquareType.EMPTY}


class Dataset:
    
    data   = None
    target = None
    
    def __init__(self,logfile='_logfile.csv'):
#         os.chdir('../') # TODO: Warning - Only test purpose
        self.filepath = logfile
#         os.chdir('./Players') # TODO: Warning - Only test purpose
        
    def load(self,generate=True): 
        exists = os.path.exists(self.filepath)
        if generate and exists: 
            os.remove(self.filepath)       
        if not exists or generate:
            # Generate new log data
            self.do_play(200,black_player=GreedyPlayer,white_player=RandomPlayer)
            self.do_play(800,black_player=RandomPlayer,white_player=RandomPlayer)
        dataset = np.loadtxt(self.filepath, delimiter=",")
        self.data     = dataset[:,:-1]
        self.target   = dataset[:,-1:].ravel() # OBS: For 1d problem
    
    def do_play(self,n,black_player=RandomPlayer,white_player=RandomPlayer):
        gambles = np.array([BatchGameGrupo1(self.filepath, black_player=black_player, white_player=white_player).play() for _ in xrange(n)])
        print "\r%s vs %s" % (black_player.name.upper(), white_player.name.upper()),
        print "Wins: %5.2f%%, Lose: %5.2f%%, Draw: %5.2f%%" % (
            100.0 * np.sum(gambles == GameStatus.BLACK_WINS.value) / n, 
            100.0 * np.sum(gambles == GameStatus.WHITE_WINS.value) / n,
            100.0 * np.sum(gambles == GameStatus.DRAW.value      ) / n,
        )
