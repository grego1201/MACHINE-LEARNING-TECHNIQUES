# -*- coding: utf-8 -*-


import codecs
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.neighbors
import numpy
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
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

#   3.2. Building the Dendrogram    
methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
selec_meth = methods[1]
criterions = ["inconsistent", "distance", "maxclust"]
sel_crit = criterions[1]

cut = 10# !!!! ad-hoc
#for meth in methods:
clusters = cluster.hierarchy.linkage(matsim, method = selec_meth)
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
cluster.hierarchy.dendrogram(clusters, color_threshold = cut)
plt.title('%s color_threshold: %d' % (selec_meth, cut))
plt.show()
#for crit in criterions:
labels = cluster.hierarchy.fcluster(clusters, cut , criterion = sel_crit)
    
    
#   4. plot
colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)
numbers = numpy.arange(len(X_pca))
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
    

#   6. characterization
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

for c in range(1,n_clusters_+1):
    print 'Group', c
    for i in range(len(data_normalizated[0])):
        column = [row[i] for j,row in enumerate(data) if labels[j] == c]
        if len(column) != 0:
            print i, numpy.mean(column)
        
        