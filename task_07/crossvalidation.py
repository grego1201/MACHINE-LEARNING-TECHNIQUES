#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:46:32 2017

@author: sergio
"""
import normalization as norm
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV    

def pearson_correlation(data, verbose = True):
   
    plt.rcdefaults()
    #----------------
    norm_data = norm.normalization_with_minmax(data)
    
    # 2. Principal Component Analysis
    estimator, X_pca = norm.pca(norm_data)
    #----------------
    
    features = data.columns.tolist()[:-1]
        # 1. Correlation
    corr = map(lambda x1: pearsonr(x1,norm_data.tolist()[-1])[0],
               [norm_data[x].tolist() for x in range(len(features))])
    
    features_corr = zip(features, corr)
    
    if verbose:
        y_pos = np.arange(len(features))
        
        plt.bar(y_pos, corr, align='center')
        plt.xticks(y_pos, features, rotation = 90)
        plt.ylabel('Correlation')
        plt.title('Correlation features vs \'total_cases\'')
        plt.show()
    
        print ''
        print tabulate(features_corr, headers = ['Feature','R value'])
    
    return features_corr


def cross_validation(data, feature_groups, features = None, target = None, verbose = False):
    data.head()

    data_adapted = data.drop(labels = ['city', 'year'], axis = 1, inplace = False)
    
    plt.rcdefaults()
    
    features_corr = pearson_correlation(data_adapted, verbose)
        
    # Select the feature with the highest correlation of each group.
    features_selected = []
    
    for group, feature_list in feature_groups.iteritems():    
        max = 0    
        candidate = []
        for f_i in feature_list:
            for _tuple in features_corr:
                if (_tuple[0] == f_i) and (_tuple[1] > max):
                    max = _tuple[1]
                    candidate = f_i
        if len(candidate)>0:            
            features_selected.append(candidate)
        
   
    X_train = data[features_selected]
    y_train = data[target]
        

    reg = DecisionTreeRegressor()
    
    # Set the parameters by cross-validation
    param_grid = {'criterion': ['mse'],
                  'max_depth': np.arange(2, 30), 'random_state':[0],
                }
    
    grid_reg = GridSearchCV(reg, param_grid, cv = 10,
                            scoring = 'neg_mean_squared_error')
    
    grid_reg.fit(X_train, y_train)
    best_max_depth = grid_reg.best_params_.get('max_depth')
    
    
    return features_selected, best_max_depth

    