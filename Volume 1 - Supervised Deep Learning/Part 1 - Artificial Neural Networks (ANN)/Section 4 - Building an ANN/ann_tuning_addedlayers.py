
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
# create the ANN
# ---------------------------------------------
#import keras that uses tensorflow
import keras
# import sequential module (initalize ANN) and dense Module (Build layers of ANN) 
from keras.models import Sequential
from keras.layers import Dense
# dropout randomly disables neurons at each iteration to reduce overfitting
# in case of overfitting apply dropout to all layers
from keras.layers import Dropout

# Initialize the ANN as a sequence of layers
# We are Predicting a Class so use classifier
classifier = Sequential()

classifier.add(Dense(10,kernel_initializer='uniform',activation='relu', input_dim=11))
#classifier.add(Dropout(rate=0.2))
classifier.add(Dense(10,kernel_initializer='uniform',activation='relu'))
#classifier.add(Dropout(rate=0.2))

#classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
#classifier.add(Dropout(rate=0.2))
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

# Compile the ANN

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])

# Connect Training Set to the ANN :: Takes its time :: Adjust batch and epochs as required
classifier.fit(X_train, y_train, batch_size=32,epochs=100)


# ---------------------------------------------
# Create ANN - Complete
# ---------------------------------------------


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Convert predictions to true or false, threshold when predict is 1 or 0 , 50%
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Take Total values that match and compare to probability given after model is run
accuracy_test = ( cm[0,0] + cm[1,1] )/ 2000

"""
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

"""
