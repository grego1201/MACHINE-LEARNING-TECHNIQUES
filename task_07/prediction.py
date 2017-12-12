#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:56:14 2017

@author: sergio
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
    
def knn_prediction(data, split_year, relevant_features, target, data_test, verbose = False):    

    data.head()
        
    train = data[data['year']< split_year]
    test = data[data['year']>= split_year]
    
    X_train = np.array(train[relevant_features])
    y_train = np.array(train[target])
    X_test = np.array(test[relevant_features])
    y_test = np.array(test[target])
    
    
    reg = KNeighborsRegressor()
        
    # Set the parameters by cross-validation
    param_grid = {'n_neighbors': np.arange(1,31),
                  'weights': ['uniform', 'distance']
                  }
    
    grid_reg = GridSearchCV(reg, param_grid, cv = 10,
                            scoring = make_scorer(r2_score))

    grid_reg.fit(X_train, y_train).predict(X_test)
    #score = r2_score(y_test, pred_training)
    score = grid_reg.score(X_test, y_test)
    
    
    best_n_neighbors = grid_reg.best_params_.get('n_neighbors')
    best_weights = grid_reg.best_params_.get('weights')



    # prediction 
    data_test[relevant_features]
    data_test.interpolate(method='linear', limit = 3, inplace = True)
    data_test.fillna(test.mean(), inplace = True)

    prediction = grid_reg.fit(data[relevant_features], data[target]).predict(data_test[relevant_features])

        
    if verbose:    
        # show prediction
        print "\nPREDICTION:"
        xx = np.stack(i for i in range(len(prediction)))
        plt.plot(xx, prediction, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (best_n_neighbors,best_weights))
        
        plt.show()
    
    return prediction, score


def rf_prediction(data, split_year, relevant_features, target, data_test, verbose = False):

    data.head()
        
    train = data[data['year']< split_year]
    test = data[data['year']>= split_year]
    
    X_train = train[relevant_features]
    y_train = train[target]
    X_test = test[relevant_features]
    y_test = test[target]

    
    max_features = len(relevant_features)
    
    reg = RandomForestRegressor()
    
    param_grid = {'n_estimators': np.arange(10,101,10),
                 'max_depth': np.arange(2, 5), 'criterion':['mae'],
                 'max_features': [max_features], 'random_state':[0]}
    
    grid_reg = GridSearchCV(reg, param_grid, cv=10, scoring = make_scorer(r2_score))
    
    grid_reg.fit(X_train, y_train).predict(X_test)
    
    score = grid_reg.score(X_test, y_test)
       
    best_n_neighbors = grid_reg.best_params_.get('n_estimators')
    best_weights = grid_reg.best_params_.get('max_depth')
    
    # prediction
    
    data_test.interpolate(method='linear', limit = 3, inplace = True)
    data_test.fillna(test.mean(), inplace = True)

    #--------------------
    
    
    #Model construction
    prediction = grid_reg.fit(data[relevant_features], data[target]).predict(data_test[relevant_features])
    
    
    if verbose:    
        # show prediction
        print "\nPREDICTION:"
        xx = np.stack(i for i in range(len(prediction)))
        plt.plot(xx, prediction, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("RandomForest (estimators = %i, max_depth = '%s')" % (best_n_neighbors,best_weights))
        
        plt.show()
    
    
    return prediction, score