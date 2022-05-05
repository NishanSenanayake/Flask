# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:08:15 2022

@author: 14198
"""

import pandas as pd
import seaborn as sns
import numpy as np


df = pd.read_csv('heart_data.csv')
print(df.head())

df=df.drop("Unnamed: 0", axis=1)


X=df.drop("heart.disease", axis=1)
y=df["heart.disease"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn import linear_model

#Create Linear Regression object
model = linear_model.LinearRegression()

#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)

import matplotlib.pyplot as plt
plt.plot(y_test,prediction_test,'ro')
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)
import pickle

pickle.dump(model, open('model.pkl', 'wb'))

model=pickle.load(open('model.pkl', 'rb'))
print(model.predict([[20.1, 56.3]]))



