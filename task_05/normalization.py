#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:00:29 2017

@author: sergio
"""

from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def normalization_with_minmax(data):
    
    min_max_scaler = preprocessing.MinMaxScaler()
        
    return min_max_scaler.fit_transform(data.dropna().values)  

def pca(data):
    estimator = PCA (n_components = 2)
    X_pca_ = estimator.fit_transform(data)
    
    return estimator, X_pca_

def pca_plots(estimator, X_pca, index):

    fig, ax = plt.subplots(1, 1, figsize = (7, 14))
    
    for i in range(len(index)):
        plt.text(X_pca[i][0], X_pca[i][1],
                 index[i]) 

    plt.title('PCA.\nEstimation ratio: {}'
              .format(estimator.explained_variance_ratio_))
    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    plt.grid(True)
    fig.tight_layout()

    plt.show()