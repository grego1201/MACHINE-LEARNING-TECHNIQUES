# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import sklearn.neighbors
from scipy import cluster
from numpy import arange
from pylab import pcolor, show, colorbar, xticks, yticks
import seaborn as sns
from tabulate import tabulate


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
        df.drop(df.index[outliers], inplace = True)
    
    return df

def correlation_plots(data, R):
    features_length = len(data.columns)
    
    
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

def main():
    # 0. Load data
    years = range(2004, 2011)
    _filter = {'city':['sj'], 'year': years}
    excludes = range(0, 4)
    
    data = load_data("../data/dengue_features_train.csv",
                     filter_parameters = _filter, excludes_features = excludes)
        
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
    
    data_aux = data
    
    
    numbers = data_aux.dropna().index
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
    data_droped.to_csv('../data/data_droped.csv', sep='\t')
    
    features = data_droped.transpose();

    #1. Normalization of the data
    
    features_norm = normalization_with_minmax(features)

    #1.2. Principal Component Analysis
    estimator, X_pca = pca(features_norm)
    plt.plot(X_pca[:,0], X_pca[:,1],'x')

    print("Variance Ratio: ", estimator.explained_variance_ratio_) 

    fig, ax = plt.subplots()
    for i in range(len(names)):
        plt.text(X_pca[i][0], X_pca[i][1], i+1) 

    plt.xlim(min(X_pca[:,0]-0.2), max(X_pca[:,0])+0.2)
    plt.ylim(min(X_pca[:,1]-0.2), max(X_pca[:,1])+0.2)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    features_name = zip(range(1,len(names)+1),names)
    print tabulate(features_name, headers = ['# ','Feature name'])
    
    
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
