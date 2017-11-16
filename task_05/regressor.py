#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:40:06 2017

@author: sergio
"""
import graphviz 
from tabulate import tabulate
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor

def tree_regressor(data, max_depth, features_selected, feature_regression, city, verbose = False):
        #3. Build the model
    
    #3.1 Model Parametrization 
    # criterion: mse mean squared error, which is equal to variance reduction as feature selection criterion
    # splitter: best/random
    # max_depth: low value avoid overfitting
    regressor = DecisionTreeRegressor(criterion = 'mse', max_depth = max_depth)

    #3.2 Model construction
    regressor.fit(data[features_selected], data[feature_regression])
    
    
    #3.3.  Model Visualization
    
    dot_data = export_graphviz(regressor, out_file=None, 
                               feature_names = features_selected,
                               filled=True, rounded=True,  
                               special_characters=True)  
    
    graph = graphviz.Source(dot_data, format = 'png')   
    graph.render('decision_tree_mse_'+city, 'images', cleanup= True) 
    graph 
    

    # 3.4 Feature Relevances
    relevances_list = zip(features_selected, regressor.feature_importances_)
    reg_values = list(regressor.feature_importances_)
    reg_values_sort = sorted(reg_values, reverse = True)
    
    max_values_feature = []
    
    for i in range(len(reg_values)):
        if (reg_values_sort[i] >= 0.1):
            pos = reg_values.index(reg_values_sort[i])
            max_values_feature.append(features_selected[pos])
        else:
            break
    
    if verbose:
        print '\n\t\t[ RELEVANCES FEATURES (MSE) ]\n'
        print tabulate(relevances_list, headers=['Feature selected', 'Relevance'])
        print '\nFeatures with more relevance: \n\t' + str(max_values_feature)
    
    return max_values_feature