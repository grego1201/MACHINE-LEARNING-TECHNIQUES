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
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor

def pearson_correlation(data, verbose = False):
   
    plt.rcdefaults()
    #----------------
    norm_data = norm.normalization_with_minmax(data)
    
    # 2. Principal Component Analysis
    estimator, X_pca = norm.pca(norm_data)
    #----------------
    
    features = data.columns.tolist()[:-1]
        # 1. Correlation
    corr_SanJuan = map(lambda x1: pearsonr(x1,norm_data.tolist()[-1])[0],
               [norm_data[x].tolist() for x in range(len(features))])
    
    features_corr = zip(features, corr_SanJuan)
    
    if verbose:
        y_pos = np.arange(len(features))
        
        plt.bar(y_pos, corr_SanJuan, align='center')
        plt.xticks(y_pos, features, rotation = 90)
        plt.ylabel('Correlation')
        plt.title('Correlation features vs target')
        plt.show()
    
        print ''
        print tabulate(features_corr, headers = ['Feature','R value'])

    #Selection of characteristics with correlation greater than 0.7.
    features_selected = [features_corr[i][0] for i in range(len(features_corr))
                            if abs(features_corr[i][1]) > 0.7]
    
    if len(features_selected) == 0:
        features_selected = [features_corr[i][0] for i in range(len(features_corr))
                            if abs(features_corr[i][1]) >= 0.41]
    
    
    return features_selected


def cross_validation(data, verbose = False):
    plt.rcdefaults()
    
    features_selected = pearson_correlation(data, verbose)
    
    total_scores = []

    for i in range(2, 30):
        regressor = DecisionTreeRegressor(criterion = 'mse', max_depth=i)
        regressor.fit(data[features_selected], data['total_cases'])
        scores = -cross_val_score(regressor, data[features_selected],
                 data['total_cases'], scoring='neg_mean_absolute_error', cv=10)
        total_scores.append((scores.mean(), scores.std()))
    
    scores_mean = [total_scores[i][0] for i in range(len(total_scores))]
    info_regression = [(i+2, total_scores[i][0], '+/- '+str(total_scores[i][1])) for i in range(len(total_scores))]
    best_max_depth = total_scores.index(min(total_scores))+2
    
    if verbose:
        plt.plot(range(2,30), scores_mean, marker='o')
        plt.xlabel('max_depth')
        plt.ylabel('cv score')
        plt.show()
        
        print ''
        print tabulate(info_regression, headers = ['Level depth', 'Mean','Standard Deviation'])
    
        print '\nBest MAX_DEPTH: %d' % (best_max_depth)
    
    return features_selected, best_max_depth
    