import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainset = pd.read_csv('training data.csv')
X_train = trainset.iloc[:,1:5].values
y_train = trainset.iloc[:,5].values

testset = pd.read_csv('test data.csv')
X_test = testset.iloc[:,1:5].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Artificial Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3 ))

classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 8, epochs = 150)

y_pred = classifier.predict(X_test)
