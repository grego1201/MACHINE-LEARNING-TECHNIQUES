#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:38:16 2015

@author: FranciscoP.Romero

Modified on Wed Sep 30 15:42:06 2017 by:    
    GONZALO PEREZ FERNANDEZ.
    GREGORIO BALDOMERO PATINO ESTEO.
    SERGIO FERNANDEZ GARCIA.
"""
#import codecs
import csv
from numpy import arange
from pylab import pcolor, show, colorbar, xticks, yticks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing 


def exclude_features(data, excludes_list):
    index_range = range(0, len(excludes_list))
    excludes_list.sort()    
    data_aux = data
    
    for i in index_range:
        data_aux.pop(excludes_list[i]-i)
        
    return data_aux

def load_data(file, city, range_years, features_not_included):
    f = open(file, 'rt')
    data = []
    heads = []
    
    #total_in = 0
    #total_out = 0
    
    try:
        reader = csv.reader(f)
        
        for row in reader:
            newcity = row[0]
            if newcity == 'city':
                heads.append(exclude_features(row, features_not_included))
                
            elif newcity == city:
                if ( int(row[1]) in range_years ):
                    
                    #if not ('' in row):
                        # No include 'city' column (feature) and 
                        # rows whit any empty feature
                    example = exclude_features(row, features_not_included)
                    data.append(example)
                    #total_in += 1
                    #else:
                        #example = 
                        #total_out += 1

    finally:
        f.close()
        
        df = pd.DataFrame.from_records(data, columns = heads)
        df = df.replace('', np.nan, regex=True)
        
        #print(total_in)
        #print(total_out)
        #print(total_in + total_out)
        
    return df


def ndvi_control(data):
    cont = 0
    complete_data = []
    
    for example in data:
        
        if '' not in example:
            float_data = map(float, example)
            
            if len(filter(lambda x: (abs(x) > 1.0), float_data[0:4])) == 0:
                complete_data.append(float_data)
                cont = cont + 1
    print cont
    
    return complete_data

def temperature_control(data):
    cont = 0
    complete_data = []
    max_temperature_celsius = 58.0
    min_temperature_celsius = -90.0
    
    for example in data:
        
        if '' not in example:
            float_data = map(float, example)
            #print float_data
            
            if len(filter(lambda x: (x > max_temperature_celsius or 
                                     x < min_temperature_celsius), 
                            float_data[0:4])) == 0:
                complete_data.append(float_data)
                #print("{0}: {1}").format(cont, complete_data[cont])
                cont = cont + 1
    print cont
    
    return complete_data


def correlation_plots(data):
    features_length = len(data.columns)
    R = data.astype(float).corr()
    
    plt.pcolormesh(R)
    pcolor(R)
    colorbar()
    yticks(arange(0,features_length),range(0,features_length))
    xticks(arange(0,features_length),range(0,features_length))
    plt.title('Pearson correlation')
    show()

    # http://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
    # Generate a mask for the upper triangle
    sns.set(style = 'white')
    mask = np.zeros_like(R, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(200, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(R, mask=mask, cmap=cmap, vmax=.8, 
                square=True, xticklabels=1, yticklabels=1,
                linewidths=.2, cbar_kws={"shrink": .2}, ax=ax)
    
    ax.set_title('Pearson correlation')
    
    return R


def normalization_with_minmax(data):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    return min_max_scaler.fit_transform(data.dropna().values)    

def pca(data):
    estimator = PCA (n_components = 2)
    X_pca_ = estimator.fit_transform(data)
    
    return estimator, X_pca_

def pca_plots(estimator, X_pca):
    
    numbers = np.arange(len(X_pca))

    fig, ax = plt.subplots(1, 1, figsize = (7, 14))

    
    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1],
                 numbers[i] + 2) 

    plt.title('PCA.\nEstimation ratio: {}'
              .format(estimator.explained_variance_ratio_))
    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    plt.grid(True)
    fig.tight_layout()

    plt.show()


def main():
    
    years = range(2004, 2011)
    excludes = range(0, 4)
    data = load_data(
            "../data/dengue_features_train.csv",
            'sj',
            years,
            excludes)
    
    correlation_plots(data)

    data_normalizated = normalization_with_minmax(data)
    estimator, X_pca = pca(data_normalizated)
    
    pca_plots(estimator, X_pca)
    
     
if __name__ == '__main__':
	main()
