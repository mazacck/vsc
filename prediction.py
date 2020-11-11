#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[5]:


# User has to enter test dataset in the function to predict based on trained model    

# Linear Regression Model
with open("LinearRegModel","rb") as f:
    lm=pickle.load(f)

def linear_pred(data):
    """ This function predicts based on the linear regression trained model"""
    y_pred = lm.predict(data)
    return y_pred

# Decision Tree Model
with open("TreeModel","rb") as f:
    tm=pickle.load(f)

def tree_pred(data):
    """ This function predicts based on the decision tree trained model"""
    y_pred = tm.predict(data)
    return y_pred

# RandomForest Model
with open("RandomForest","rb") as f:
    rf=pickle.load(f)

def rf_pred(data):
    """ This function predicts based on the random forest trained model"""
    y_pred = rf.predict(data)
    return y_pred

# GridSearch Model
with open("GridSearch","rb") as f:
    gs=pickle.load(f)

def gs_pred(data):
    """ This function predicts based on the grid search trained model"""
    y_pred = gs.predict(data)
    return y_pred

