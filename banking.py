# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 23:49:51 2017

@author: Mukesh Gupta
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset =pd.read_csv('marketing-data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,16].values
#encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])       
X[:,2]=labelencoder.fit_transform(X[:,2])
X[:,3]=labelencoder.fit_transform(X[:,3])
X[:,4]=labelencoder.fit_transform(X[:,4])
X[:,6]=labelencoder.fit_transform(X[:,6])
X[:,7]=labelencoder.fit_transform(X[:,7])
X[:,8]=labelencoder.fit_transform(X[:,8])
X[:,10]=labelencoder.fit_transform(X[:,10])
X[:,15]=labelencoder.fit_transform(X[:,15])
    
   
labelencoder_y=LabelEncoder()
Y= labelencoder_y.fit_transform(Y)

# splitting dataset inti trainning set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.50,random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#feature selection using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',C=1000.0,gamma=0.01, random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
