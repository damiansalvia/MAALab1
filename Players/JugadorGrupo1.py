# -*- coding: utf-8 -*-
from Player import Player

from sklearn.externals import joblib
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from UtilsGrupo1 import do_move, get_vector
from Dataset import Dataset

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
            dataset.load(generate=False)
            X,y = dataset.data, dataset.target
            
            # Fit the classifier with the training data
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
         