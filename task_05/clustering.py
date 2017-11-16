#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:46:26 2017

@author: sergio
"""
import normalization as norm
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors
from scipy import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd

"""

"""
def hierarchical_clustering(data):
    
    #Normalization Data
    no_nan_data = data.dropna(how = 'any')
    norm_data = norm.normalization_with_minmax(data)
    #norm_dataIquitos = normalization_with_minmax(dataIquitos)
    
    estimator, X_pca = norm.pca(norm_data)
    #norm.pca_plots(estimator, X_pca, no_nan_data.index.values)
    
    # 1. Hierarchical Clustering
    #   1.1. Compute the similarity matrix
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(X_pca)
    avSim = np.average(matsim)
    print "%s\t%6.2f" % ('Average Distance', avSim)

    #   1.2. Building the Dendrogram    
    methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    selec_meth = methods[0]
    criterions = ["inconsistent", "distance", "maxclust"]
    sel_crit = criterions[1]

    cut = 3.7 # ad-hoc
    
    clusters = cluster.hierarchy.linkage(matsim, method = selec_meth)
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    plt.figure(figsize = (10,10));
    cluster.hierarchy.dendrogram(clusters, color_threshold = 7)
    plt.title('%s color_threshold: %d' % (selec_meth, 7))
    plt.show()
    
    labels = cluster.hierarchy.fcluster(clusters, cut , criterion = sel_crit)
    
    #   4. plot
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    numbers = no_nan_data.index.values
    n_total = len(numbers)
    
    fig, ax = plt.subplots()
        
    for i in range(n_total):
        plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
    
    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    ax.grid(True)
    fig.tight_layout()
    plt.title('Method: %s, Cut: %d, Criterion: %s' % (selec_meth, cut, sel_crit))
    plt.show()
    
    # 5. characterization
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    
    
    groups = {}
    outliers = []
    for c in range(1, n_clusters_+1):
        
        elements = []
        
        for i in range(len(norm_data)):
            if labels[i] == c:
                elements.append(numbers[i])
                
        groups[c] = elements
        
        n_elements = len(elements)
        percent = (float(n_elements)/float(n_total))*100.0
        
        if (percent<=2.0):
            outliers.append(elements)
        elif (percent <= 10.0):
            
            df = pd.DataFrame(data, index = elements)
            
            sub_elements, sub_outliers = hierarchical_clustering(df)
            if sub_outliers != None:
                outliers.append(sub_outliers)
            elements = sub_elements
            n_elements = len(elements)
        
        
        print '\nGroup %2d, length: %d, total %d, percent %5.2f ' % (c, n_elements, n_total, percent)
        #print elements
    print outliers
        
    if (len(outliers) == 0):
        outliers = None
    
    return groups, outliers

"""

"""
def hierarchical_clustering_features(data):
    
    names = list(data)
    
    data_aux = data
    data_droped = data_aux.dropna()
    
    data_transpose = data_droped.transpose();

    #1. Normalization of the data
    data_transpose_norm = norm.normalization_with_minmax(data_transpose)

    #1.2. Principal Component Analysis
    estimator, X_pca = norm.pca(data_transpose_norm)
    #plt.plot(X_pca[:,0], X_pca[:,1],'x')

    #print("Variance Ratio: ", estimator.explained_variance_ratio_) 
    """
    fig, ax = plt.subplots()
    
    for i in range(len(data_transpose)):
        plt.text(X_pca[i][0], X_pca[i][1], i+1) 

    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    """
    data_transpose_name = zip(range(1,len(names)+1),names)
    print tabulate(data_transpose_name, headers = ['# ','Feature name'])
    
    
    # 2. Compute the similarity matrix
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(data_transpose_norm)
    avSim = np.average(matsim)
    print "%s\t%6.2f" % ('Average Distance', avSim)
    
    
    # 3. Building the Dendrogram	
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    cluster.hierarchy.dendrogram(clusters, color_threshold = 5, labels = names, leaf_rotation=90)
    plt.show()
    
"""

"""
def kMeans_clustering(data):

    # 1. Data normalization
    no_nan_data = data.dropna(how = 'any')
    norm_data = norm.normalization_with_minmax(data)
    
    # 2. Principal Component Analysis
    estimator, X_pca = norm.pca(norm_data)
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
        labels = km.fit_predict(norm_data)
        distortions.append(km.inertia_)
        silhouettes.append(metrics.silhouette_score(norm_data, labels))
    
    """
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
    """
      
    # Set K value
    k = distortions.index(max(distortions)) + 2
    print '\n\n K value is: ' + str(k) + '\n\n'


    ### 5. Execute clustering 
    km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(norm_data)

    

    ### 6. Plot the results
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
    plt.grid()
    plt.show()    
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    numbers = no_nan_data.index.values
    n_total = len(numbers)
    
    groups = {}
    outliers = []
    for c in range(1, n_clusters_+1):
        
        elements = []
        
        for i in range(len(norm_data)):
            if labels[i] == c-1:
                elements.append(numbers[i])
                
        groups[c] = elements
        
        n_elements = len(elements)
        percent = (float(n_elements)/float(n_total))*100.0
        
        if (percent<=2.0):
            outliers.append(elements)
        
        print '\nGroup %2d, length: %d, total %d, percent %5.2f ' % (c, n_elements, n_total, percent)
        
    if (len(outliers) == 0):
        outliers = None
    
    return groups, outliers