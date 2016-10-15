

dataset = Dataset()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=0)

algo = [
    {},
]


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


# Fit the classifier with the training data
self.model.fit(X_train, y_train)

# Evaluate the model
evaluate(self.model,X_test,y_test)

# Dump the classifier
joblib.dump(self.model, '%s.pkl'  % self.name) # TODO: Descomentar