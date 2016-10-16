# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from DataTypes import SquareType

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


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
        Agent,Opponent = np.sum(matrix == color.value), np.sum(matrix == OTHER[color].value)
        if Agent == Opponent: 
            vector = [Ia,Io,0,0] # Draw board
        elif Agent > Opponent:    
            vector = [Ia,Io,Imax,0] # Wins board
        else: 
            vector = [Ia,Io,0,Imax] # Defeat board
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


def plot_3d_barchart(values,total,xlabels,title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    possibles = ['wins', 'draw', 'lose']
    color = {"wins":"g","draw":"y","lose":"r"}
    
    for case, z in zip(possibles, [10, 5, 0]):
        xs = np.arange(len(values[case]))
        ys = values[case]
        cs = [color[case]] * len(xs)
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8, align='center')
    
    ax.set_xlim([-1,total])
    ax.set_ylim([-5,15])
    ax.set_autoscaley_on(False)
#     plt.gca().xaxis.set_major_formatter(FormatStrFormatter('Caso %d'))
    ax.set_xticklabels(xlabels)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='center', va='center')
    for label in ax.get_xticklabels():
        label.set_fontsize(8)
    plt.xticks(np.arange(-1, total, 1.0))
    ax.set_yticklabels([])
    ax.set_zlabel('%')
    ax.set_title(title)
    
    plt.tight_layout()
    fig.savefig('_plot_%s.png'%title.replace(" ","_"), bbox_inches='tight')
    plt.show()
    
    
    
if __name__ == '__main__':
    import csv
    plot_greedy = {"wins":[],"draw":[],"lose":[]}
    plot_random = {"wins":[],"draw":[],"lose":[]}
    with open('_results.csv','r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count:
                data = row[:3] # Take the firts three columns
                plot_random['wins'].append(data[0])
                plot_random['draw'].append(data[1])
                plot_random['lose'].append(data[2])
                data = row[3:]          
                plot_greedy['wins'].append(data[0])
                plot_greedy['draw'].append(data[1])
                plot_greedy['lose'].append(data[2])
            count += 1
    total = len(plot_greedy['wins'])
    xlabels = ["Case %i"%i for i in xrange(total)]
    plot_3d_barchart(plot_greedy, total, xlabels, title="JUGADORGRUPO1 vs. GREEDYPLAYER")
    plot_3d_barchart(plot_random, total, xlabels, title="JUGADORGRUPO1 vs. RANDOMPLAYER")
    
        