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
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd

"""

"""
def hierarchical_clustering(data, cut = None, first_total = None, verbose = False):
    
    #Normalization Data
        
    
    norm_data = norm.normalization_with_minmax(data)
    
    estimator, X_pca = norm.pca(norm_data)
     
    # 1. Hierarchical Clustering
    #   1.1. Compute the similarity matrix
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(X_pca)
    avSim = np.average(matsim)
    
    if verbose:
        print "%s\t%6.2f" % ('Average Distance', avSim)

    #   1.2. Building the Dendrogram    
    methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    selec_meth = methods[0]
    criterions = ["inconsistent", "distance", "maxclust"]
    sel_crit = criterions[1]

    #cut = 3.7 # ad-hoc
    
    clusters = cluster.hierarchy.linkage(matsim, method = selec_meth)
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    if verbose:
        plt.figure(figsize = (10,10));
    
    dendrogram_data=cluster.hierarchy.dendrogram(clusters,no_plot= (not verbose), color_threshold = 7)
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    dendrogram_flat = flatten(dendrogram_data.get('dcoord'))
    
    if not cut:
        cut = max(dendrogram_flat)
        dendrogram_flat.remove(cut)
        max_distance = cut - 0.1
        
    else:
        max_distance = max(dendrogram_flat)-0.1
    
   
    if verbose:
        plt.title('%s color_threshold: %d' % (selec_meth, 7))
        plt.show()
    
    labels = cluster.hierarchy.fcluster(clusters, max_distance , criterion = sel_crit)
    
    #   4. plot
    numbers = data.index.values
    n_total = len(numbers)
    
    if not first_total:
        first_total = n_total
    
    if verbose:
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)
        fig, ax = plt.subplots()
        
        for i in range(n_total):
            plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
        
        plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
        plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
        ax.grid(True)
        fig.tight_layout()
        plt.title('Method: %s, Cut: %5.2f, Criterion: %s' % (selec_meth, max_distance, sel_crit))
        plt.show()
    
    # 5. characterization
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if verbose:
        print('Estimated number of clusters: %d' % n_clusters_)
    
    #print 'cut: %.2f' % cut
    groups = {}
    outliers = []
    cut_percent = cut *(1 - avSim + 0.2)
    max_cut = cut - cut_percent
    
    for c in range(1, n_clusters_ + 1):
        
        elements = []
        
        for i in range(len(norm_data)):
            if labels[i] == c:
                elements.append(numbers[i])
                
        groups[c] = elements
        
        n_elements = len(elements)
        percent = (float(n_elements+(first_total-n_total))/float(n_total))*100.0
        
        if (percent <= 2.0 and max_cut <= max_distance):
            outliers.append(elements)
            del groups[c]
            
            if verbose:
                print 'To delete 2%: \t'
                print elements
        
        elif (percent <= 10.0 and max_cut <= max_distance):
            
            df = pd.DataFrame(data, index = elements)
            
            sub_elements, sub_outliers, cut = hierarchical_clustering(df, 
                                                        cut = cut,
                                                        first_total = first_total,
                                                        verbose = verbose)

            if verbose and sub_outliers:
                print 'To delete 10%: \t'
                print sub_outliers
            
            if sub_outliers:
                outliers.append(sub_outliers)
            else:
                outliers = None
                    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    if outliers:
        final_outliers = flatten(outliers)
    else:
        final_outliers = None
        
    return groups, final_outliers, cut
    
"""

"""
def hierarchical_clustering_features(data, verbose = True):
    
    names = list(data)
    
    data_transpose = data.transpose();

    #1. Normalization of the data
    data_transpose_norm =  norm.normalization_with_minmax(data_transpose)

    #1.2. Principal Component Analysis
    estimator, X_pca = norm.pca(data_transpose_norm)
    #------------------------

    if verbose:
        plt.plot(X_pca[:,0], X_pca[:,1],'x')
        print("Variance Ratio: ", estimator.explained_variance_ratio_) 
  
        fig, ax = plt.subplots()
    
        for i in range(len(data_transpose)):
            plt.text(X_pca[i][0], X_pca[i][1], i+1) 
    
        plt.xlim(min(X_pca[: , 0] - 0.2), max(X_pca[: , 0]) + 0.2)
        plt.ylim(min(X_pca[: , 1] - 0.2), max(X_pca[: , 1]) + 0.2)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
    
        data_transpose_name = zip(range(1,len(names) + 1), names)
        print tabulate(data_transpose_name, headers = ['# ','Feature name'])
    
    # 2. Compute the similarity matrix
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(data_transpose_norm)
    avSim = np.average(matsim)
    clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
    labels = cluster.hierarchy.fcluster(clusters ,6, criterion = 'maxclust')
   
    
    feature_name = data_transpose.index.values

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    groups = {}
    
    for c in range(1, n_clusters_+1):
        
        
        elements = []
        
        for i in range(len(data_transpose_norm)):
            if labels[i] == c:
                elements.append(feature_name[i])
                
        groups[c] = elements
        
    if verbose:
        print "%s\t%6.2f" % ('Average Distance', avSim)
        plt.figure(figsize = (7,7));
        # 3. Building the Dendrogram	
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage    
        # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
        cluster.hierarchy.dendrogram(clusters, color_threshold = 5, labels = names, leaf_rotation=90)
        print('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    return groups
