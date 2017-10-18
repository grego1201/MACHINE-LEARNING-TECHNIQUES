# -*- coding: utf-8 -*-


import codecs
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np



## Function for load data except Outliers
def load_data(file, outliers):
    f = codecs.open(file, "r", "utf-8")
    features_name = []
    features_data = []
    count = 0
    for line in f:
        if (count > 0 and (count not in outliers)): 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		
		if row != []:
			features_data.append(map(float, row))
        else:
           features_name = line.replace ('"', '').split(",")
        count += 1
 
   
    return features_data, features_name

## Function for nomralization with MinMax algorithm
def normalization_with_MinMax(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    
    return  min_max_scaler.fit_transform(data)    




# 0. Load data.
    # The outliers are 21 and 38 
data, labels = load_data("../data/dengue_pivot_mean.csv", [20, 37])

# 1. Data normalization
data_normalizated = normalization_with_MinMax(data)

# 2. Principal Component Analysis
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data_normalizated)

plt.plot(X_pca[:,0], X_pca[:,1],'x')

# 3. Setting parameters (ad-hoc)

# parameters
methods = ["random", "k-means++"]
init = methods[0] # initialization method 
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

