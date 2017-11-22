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
#from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

def calc_distance(data):
    result = []
    values =[]

    for key, value in data.iteritems():
        values.append(value)
    
    n_position = [values[0][i][0] for i in range(len(values[0]))]
    
    for i in range(len(n_position)):
        result.append((n_position[i], abs(values[0][i][1]-values[1][i][1])))
    
#    for i,j in values[1], values[2]:
        #Restar los valores y pasarlo a valor absoluto
    
    #result = [lambda]               
    return result

def calc_minimal(data):
    #uniform_distance = calc_distance(data)
    
    uniform = [data.get('uniform')[i][1] for i in range(len(data.get('uniform')))]
    n_position = [data.get('uniform')[i][0] for i in range(len(uniform))]
    
    #distance = [data.get('distance')[i][1] for i in range(len(uniform_distance))]
    
    sorted_uniform = sorted(uniform)
    
    best_neighbours=[]
    
    for i in range(len(uniform)):
        position = uniform.index(sorted_uniform[i])
        best_neighbours.append(n_position[position])
    
    return best_neighbours
    
    

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

    #Selection of features with correlation greater than 0.7.
    features_selected = [features_corr[i][0] for i in range(len(features_corr))
                            if abs(features_corr[i][1]) > 0.7]
    
    #Else, select features with high correlation.
    if len(features_selected) == 0:
        features_selected = [features_corr[i][0] for i in range(len(features_corr))
                            if abs(features_corr[i][1]) >= 0.41]
    
    
    return features_selected


def cross_validation(data, algorithm=None, features = None, target = None, verbose = False):

    
    
    
    if (algorithm == 'DecisionTreeRegressor'):
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
    
    elif (algorithm == 'KNN'):
        
        data.head()
        X = data[features]
        y = data[target]
        
        data_weights = { }
        
        for i, weights in enumerate(['uniform', 'distance']):
            total_scores = []
            for n_neighbors in range(1,30):
                knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                knn.fit(X,y)
                scores = -cross_val_score(knn, X,y, 
                                            scoring='neg_mean_absolute_error', cv=10)
                total_scores.append((n_neighbors, scores.mean()))
            
                data_weights[weights] = total_scores
                
            if verbose:
                scores_mean = [total_scores[j][1] for j in range(len(total_scores))]
                plt.plot(range(0,len(scores_mean)), scores_mean, 
                         marker='o', label=weights)
                plt.ylabel('cv score')
        
        best_neighbors = calc_minimal(data_weights)
        
        if verbose:
            plt.legend()
            plt.show()   
        
        
        
                #1. Build the model
        
        xx = np.stack(i for i in range(len(y)))
        
        n_neighbors = best_neighbors[0] # BEST PARAMETER
        
        plt.figure(figsize = (10,10));
        
        for i, weights in enumerate(['uniform', 'distance']):
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            y_pred = knn.fit(X,y).predict(X)
            
            plt.subplot(2, 1, i + 1)
            plt.plot(xx, y, c='k', label='data')
            plt.plot(xx, y_pred, c='g', label='prediction')
            #plt.axis('tight')
            plt.legend()
            plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                        weights))
        
        plt.show()
                
        return n_neighbors, X, y
    