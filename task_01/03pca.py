# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 17:35:44 2015

@author: FranciscoP.Romero


Modified on Wed Sep 27 18:38:27 2017 by:    
    GONZALO PEREZ FERNANDEZ.
    GREGORIO BALDOMERO PATINO ESTEO.
    SERGIO FERNANDEZ GARCIA.
    
"""
import codecs
import matplotlib.pyplot as plt
import numpy
from sklearn.decomposition import PCA
from sklearn import preprocessing 

# 0. Load Data
f1 = codecs.open("data/dengue_pivot_select.csv", "r", "utf-8")
weeks_selected = []
count_select = 0
for line in f1:
	if count_select > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		row.pop(0)
		if row != []:
			data = [float(el) for el in row]
			weeks_selected.append(data)
	count_select += 1
    
    
f = codecs.open("data/dengue_pivot_mean.csv", "r", "utf-8")
weeks_means = []
count_means = 0
for line in f:
	if count_means > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		row.pop(0)
		if row != []:
			data = [float(el) for el in row]
			weeks_means.append(data)
	count_means += 1

#1. Normalization of the data
min_max_scaler_from_selected = preprocessing.MinMaxScaler()
weeks_selected = min_max_scaler_from_selected.fit_transform(weeks_selected)
       
min_max_scaler_from_means = preprocessing.MinMaxScaler()
weeks_means = min_max_scaler_from_means.fit_transform(weeks_means)
#2. PCA Estimation
estimator_from_selected = PCA (n_components = 2)
X_pca_from_selected = estimator_from_selected.fit_transform(weeks_selected)

estimator_from_means = PCA (n_components = 2)
X_pca_from_means = estimator_from_means.fit_transform(weeks_means)

print(estimator_from_selected.explained_variance_ratio_) 
print(estimator_from_means.explained_variance_ratio_) 
#3.  plot 
numbers_01 = numpy.arange(len(X_pca_from_selected))
numbers_02 = numpy.arange(len(X_pca_from_means))

fig, ax = plt.subplots(2, 1, figsize = (10, 20))

plt.subplot(2,1,1)
for i in range(len(X_pca_from_selected)):
    plt.text(X_pca_from_selected[i][0], X_pca_from_selected[i][1], numbers_01[i] + 2) 

plt.title('PCA suppresing empty data')
plt.xlim(-1.2, 2)
plt.ylim(-0.7, 1.7)
ax[0].grid(True)
fig.tight_layout()

plt.subplot(2,1,2)
for i in range(len(X_pca_from_means)):
    plt.text(X_pca_from_means[i][0], X_pca_from_means[i][1], numbers_02[i] + 2) 
plt.title('PCA with arithmetic means')
plt.xlim(-1.2, 2)
plt.ylim(-0.7, 1.7)
ax[1].grid(True)
fig.tight_layout()

plt.show()

