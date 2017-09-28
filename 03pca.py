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
f = codecs.open("data/dengue_pivot.csv", "r", "utf-8")
weeks = []
count = 0
for line in f:
	if count > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		row.pop(0)
		if row != []:
			data = [float(el) for el in row]
			weeks.append(data)
	count += 1

#1. Normalization of the data
min_max_scaler = preprocessing.MinMaxScaler()
weeks = min_max_scaler.fit_transform(weeks)
       
#2. PCA Estimation
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(weeks)

print(estimator.explained_variance_ratio_) 

#3.  plot 
numbers = numpy.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    #Se suma dos para poder ver que estado es en el fichero
    plt.text(X_pca[i][0], X_pca[i][1], numbers[i] + 2) 
#plt.xlim(-1, 4)
#plt.ylim(-0.2, 1)
ax.grid(True)
fig.tight_layout()
plt.show()

