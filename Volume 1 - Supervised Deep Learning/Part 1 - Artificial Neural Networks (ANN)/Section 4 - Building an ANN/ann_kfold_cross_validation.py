# Pre-Requisites
# pip install theanos
# pip install tensorflow
# pip install keras
# install sypder
# sudo pip3 install spyder3
#  spyder3 ann1.py

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  #upperbound excluded so it is 13 not 12
y = dataset.iloc[:, 13].values # exited last column

# ---------------------------------------------
# Pre Processing Stage
# ---------------------------------------------

# Encoding categorical Independent Variables
# Turn non-numeric values to numeric values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encode Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Encode Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Create DUmmy Variables for Country
# Created 3 colums of dummy variables corresponding to country
# Columns 0:2
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Remove first column of dummy variables to not fall into dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------------------------------
# Pre Processing Complete
# ---------------------------------------------

# --------------------
# Evaluating the ANN
# --------------------

# Tricky thing we used keras for the model,
# and k-fold cross validation belongs to scikit learn
# Someway combine them together (Keras wrapper  for scikit learn)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    # this classifier is local to the function
    localclassifier = Sequential()
    localclassifier.add(Dense(10,kernel_initializer='uniform',activation='relu', input_dim=11))
   # localclassifier.add(Dropout(rate=0.1))
    localclassifier.add(Dense(10,kernel_initializer='uniform',activation='relu'))
   # localclassifier.add(Dropout(rate=0.1))
    localclassifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    localclassifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
    return localclassifier
    
# Create global classifier that trains with k fold cross validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size=32,epochs=500)
#cv is the number of kfold cross validations that is applied, 10 is a good number
#n_jobs is the number of cpus to use with (-1) to use all cpus
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 20)
mean = accuracies.mean()
varriance = accuracies.std()