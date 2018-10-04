
# ---------------------------------------------
# Pre Processing Stage
# ---------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  #upperbound excluded so it is 13 not 12
y = dataset.iloc[:, 13].values # exited last column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------------------------------
# Pre Processing Complete
# ---------------------------------------------

# ---------------------------------------------
# Tuning the ANN
# ---------------------------------------------

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    localclassifier = Sequential()
    localclassifier.add(Dense(6,kernel_initializer='uniform',activation='relu', input_dim=11))
    localclassifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    localclassifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    localclassifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return localclassifier
    
classifier = KerasClassifier(build_fn = build_classifier)
# Tune the Epochs and Batch Size Values
# Create Dictionary
# Each key is the hyper-paramet we want to input
# Values of the Keys are the values we want to try
# All of these will be combined, then GridSearch will train with these
parameters = {'batch_size': [25, 32],
              'epochs': [10, 20],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# fit grid search to the training
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy =  grid_search.best_score_


