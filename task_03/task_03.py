# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing 
from sklearn.cluster import KMeans
from sklearn import metrics


def load_data(file, filter_parameters = None, excludes_features = None,
              outliers = None):
    
    df = pd.read_csv(file)
    
    df.replace('', np.nan, inplace=True, regex=True)
  
    if filter_parameters != None:
        for key, value in filter_parameters.iteritems():
            df_aux = df.loc[df[key].isin(value)]
            df=df_aux
    
    if excludes_features != None:
        if type(excludes_features)==list and type(excludes_features[0])== str:
            df.drop(labels = excludes_features, axis = 1, inplace = True)
        elif type(excludes_features)==list and type(excludes_features[0])== int:
            df.drop(df.columns[excludes_features], axis = 1, inplace = True)
            
    if outliers != None:
        df.drop(outliers, axis = 0, inplace = True)
    
    return df


def normalization_with_minmax(data):
    
    min_max_scaler = preprocessing.MinMaxScaler()
        
    return min_max_scaler.fit_transform(data.dropna().values)    


def pca(data):
    estimator = PCA (n_components = 2)
    X_pca_ = estimator.fit_transform(data)
    
    return estimator, X_pca_


def main():
    # 0. Load data
    years = range(2004, 2011)
    _filter = {'city':['sj'], 'year': years}
    excludes = range(0, 4)
    _outliers = [732,769]
    data = load_data(
            "../data/dengue_features_train.csv",
            filter_parameters = _filter, excludes_features = excludes,
              outliers = _outliers)
    
    
    
    # 1. Data normalization
    data_normalizated = normalization_with_minmax(data)

    # 2. Principal Component Analysis
    estimator, X_pca = pca(data_normalizated)
    #plt.plot(X_pca[:,0], X_pca[:,1],'x')
    
    
    # 3. Setting parameters (ad-hoc)
    
    # parameters
    
    init = 'k-means++' # initialization method 
    iterations = 10 # to run 10 times with different random centroids to choose the final model as the one with the lowest SSE
    max_iter = 300 # maximum number of iterations for each single run
    tol = 1e-04 # controls the tolerance with regard to the changes in the within-cluster sum-squared-error to declare convergence
    random_state = 0 # random
    
    
    distortions = []
    silhouettes = []
    
    for i in range(2, 11):
        km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
        labels = km.fit_predict(data_normalizated)
        distortions.append(km.inertia_)
        silhouettes.append(metrics.silhouette_score(data_normalizated, labels))
    
    # 4. Plot results to know which K set
    # Plot distoritions
    plt.plot(range(2,11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

    # Plot Silhouette
    plt.plot(range(2,11), silhouettes , marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silohouette')
    plt.show()

      
    # Set K value
    k = 2


    ### 5. Execute clustering 
    km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(data_normalizated)


    ### 6. Plot the results
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
    plt.grid()
    plt.show()


if __name__ == '__main__':
	main()