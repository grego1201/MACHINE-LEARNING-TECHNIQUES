# -*- coding: utf-8 -*-


import codecs
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.neighbors
import numpy as np
from scipy import cluster



def load_data(file):            
    f = codecs.open(file, "r", "utf-8")
    features_name = []
    features_data = []
    count = 0
    for line in f:
        if count > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		
		if row != []:
			features_data.append(map(float, row))
        else:
           features_name = line.replace ('"', '').split(",")
        count += 1
 
   
    return features_data, features_name

#http://scikit-learn.org/stable/modules/preprocessing.html
def normalization_with_MinMax(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    
    return  min_max_scaler.fit_transform(data)    


# 0. Load data
data, names = load_data("../data/dengue_pivot_mean.csv")

features = np.transpose(data);

# 1. Data normalization
data_normalizated = normalization_with_MinMax(data)

# 2. Principal Component Analysis
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data_normalizated)

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

cut = 10# !!!! ad-hoc

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


# -----------------------------------------------------------------------------
features = np.transpose(data);

#1. Normalization of the data
features_norm = normalization_with_MinMax(features)

#1.2. Principal Component Analysis
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(features_norm)
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