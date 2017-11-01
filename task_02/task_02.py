# -*- coding: utf-8 -*-

import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import sklearn.neighbors
from scipy import cluster


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
    excludes = range(0, 4)
    data = load_data(
            "../data/dengue_features_train.csv",
            'sj',
            years,
            excludes)

    
    # 1. Data normalization
    data_normalizated = normalization_with_minmax(data)

    # 2. Principal Component Analysis
    estimator, X_pca = pca(data_normalizated)
    plt.plot(X_pca[:,0], X_pca[:,1],'x')

    
    # 3. Hierarchical Clustering
    #   3.1. Compute the similarity matrix
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(X_pca)
    avSim = np.average(matsim)
    print "%s\t%6.2f" % ('Average Distance', avSim)

    #   3.2. Building the Dendrogram    
    methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    selec_meth = methods[1]
    criterions = ["inconsistent", "distance", "maxclust"]
    sel_crit = criterions[1]

    cut = 10 # ad-hoc

    clusters = cluster.hierarchy.linkage(matsim, method = selec_meth)
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    cluster.hierarchy.dendrogram(clusters, color_threshold = 7)
    plt.title('%s color_threshold: %d' % (selec_meth, 7))
    plt.show()

    labels = cluster.hierarchy.fcluster(clusters, cut , criterion = sel_crit)
    
    
    #   4. plot
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    numbers = np.arange(len(X_pca))
    fig, ax = plt.subplots()
        
    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
    
    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    ax.grid(True)
    fig.tight_layout()
    plt.title('Method: %s, Cut: %d, Criterion: %s' % (selec_meth, cut, sel_crit))
    plt.show()
    print 'Number of clusters %d %s' % (len(set(labels)), sel_crit)

    
    # -------------------------------------------------------------------------
    
    names = list(data)
    
    data_aux = data
    data_droped = data_aux.dropna()
    
    features = data_droped.transpose();

    #1. Normalization of the data
    
    features_norm = normalization_with_minmax(features)

    #1.2. Principal Component Analysis
    estimator, X_pca = pca(features_norm)
    plt.plot(X_pca[:,0], X_pca[:,1],'x')

    print("Variance Ratio: ", estimator.explained_variance_ratio_) 

    fig, ax = plt.subplots()
    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], names[i]) 

    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    # 2. Compute the similarity matrix
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(features_norm)
    avSim = np.average(matsim)
    print "%s\t%6.2f" % ('Average Distance', avSim)

    # 3. Building the Dendrogram	
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    cluster.hierarchy.dendrogram(clusters, color_threshold = 3, labels = names, leaf_rotation=90)
    plt.show()

if __name__ == '__main__':
	main()