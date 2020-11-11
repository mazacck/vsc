#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor


# In[3]:


import pickle


# In[ ]:


# Linear Regression Model
with open("LinearRegModel","rb") as f:
    lm=pickle.load(f)

def linear_pred(Xtrain,Ytrain):
    """ This function scores based on the linear regression trained model """
    y_pred = lm.predict(Xtrain)
    lin_mse = mean_squared_error(Ytrain, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Mean squared error is {}".format(lin_rmse))

# Decision Tree Model
with open("treeModel","rb") as f:
    tm=pickle.load(f)

def tree_pred(Xtrain,Ytrain):
    """ This function scores based on the decision tree trained model"""
    y_pred = tm.predict(Xtrain)
    lin_mse = mean_squared_error(Ytrain, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Mean squared error is {}".format(lin_rmse))

# RandomForest Model
with open("RandomForest","rb") as f:
    rf=pickle.load(f)

def rf_pred(Xtest,Ytest):
    """ This function scores based on the random forest trained model"""
    y_pred = rf.predict(Xtrain)
    lin_mse = mean_squared_error(Ytrain, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Mean squared error is {}".format(lin_rmse))

# GridSearch Model
with open("GridSearch","rb") as f:
    gs=pickle.load(f)

def gs_pred(Xdata,Ydata):
    """ This function scores based on the grid search trained model"""
    y_pred = gs.predict(Xtrain)
    lin_mse = mean_squared_error(Ytrain, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Mean squared error is {}".format(lin_rmse))

