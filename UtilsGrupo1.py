# -*- coding: utf-8 -*-

import numpy as np
import itertools
from copy import deepcopy

from DataTypes import SquareType

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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


def plot_3d_barchart(values,title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    color = {"wins":"g","draw":"y","lose":"r"}
    
    for case, z in zip(['wins', 'draw', 'lose'], [30, 20, 10]):
        xs = np.arange(len(values[case]))
        ys = values[case]
    
        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [color[case]] * len(xs)
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Result')
    ax.set_zlabel('%')
    
    ax.set_title(title)
    
    fig.savefig('%s.png'%title, bbox_inches='tight')
    
    plt.show()