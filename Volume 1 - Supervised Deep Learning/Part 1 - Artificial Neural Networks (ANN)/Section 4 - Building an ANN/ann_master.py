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

# ---------------------------------------------
# Create ANN - Stage
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

#STochastic Gradient Descent
# Step 1: Randomly initialize weights - Dense takes care of this
# Step 2: Input the first observation/row of your data into the input layer (11 Variables)
# Step 3: Forward Propagation until get value of y hat
# CHoose  ACtivation Function for input layers : Using Rectifier
# Choose Activation Function for output layer:  USing Sigmoid will give probabilities
# Step 4: Compare y hat to y
# Step 5: Back Propagation - update weights 
# STep 6: Repeat Step 1-5 and update weights after each observation or group of observation
# STep 7: After all training data processed: Repeat (Called Epoch)

# Add input and first hidden layer (Choose # of nodes in hidden layer)
# Average Number of nodes in input and output layer = # of nodes in hidden layer
# Or use parameter tuning to determine # of nodes

# Create the First Hidden Layer -input_dim=11 expect 11 inputs
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu', input_dim=11))
# rate: fraction of neurons, (recommended to keep under 0.5 50%)
classifier.add(Dropout(rate=0.1))
# Add more HIdden Layers (Deep learning uses many hidden layers) -remove input
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.1))
# Add final output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

# Compile the ANN
# OPtimizer to find the best value for weights - 'adam' a stochastic gradient descent algo
# loss function - logaritmic loss function. 
# if depedent variable is binary loss function called binary_crossentropy 
# if dependent variable is more than 2 loss function called categorical_crossentropy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Connect Training Set to the ANN :: Takes its time :: Adjust batch and epochs as required
classifier.fit(X_train, y_train, batch_size=5,epochs=10)


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

""" Predict the following customer 
 Geography: France - equals 0,0 of dummy variable
 Credit Score: 600
 Gender: Male - equals 1
 Age: 40 years old
 Tenure: 3 years
 Balance: $60000
 Number of Products: 2
 Does this customer have a credit card ? Yes
 Is this customer an Active Member: Yes
 Estimated Salary: $50000 """
 
 # Horizontal vector for np.array to match rows; use double pair of brackets
 new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
# predection needs to be on the same scale as applied to training set
 # sc was fitted to training set so you can apply to another set with same scale
new_pred = (new_pred > 0.5)


""" Trainining against one training set you get  variable accuracy values.
THis is the Bias-Variance Tradeoff :: 
    HIgh Bias Low Variance, 
    Low Bias Low Variance, same as (Good Accuracy/Small Variance)
    High Bias High Variance,
    Low Bias High Variance, same as (Good Accuracy/Large Variance)
k-fold cross validation will help with this. """

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

def build_classifier():
    # this classifier is local to the function
    localclassifier = Sequential()
    localclassifier.add(Dense(6,kernel_initializer='uniform',activation='relu', input_dim=11))
    localclassifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    localclassifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    localclassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return localclassifier
    
# Create global classifier that trains with k fold cross validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size=5,epochs=10)
#cv is the number of kfold cross validations that is applied, 10 is a good number
#n_jobs is the number of cpus to use with (-1) to use all cpus
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
varriance = accuracies.std()