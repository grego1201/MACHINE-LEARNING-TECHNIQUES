# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:38:16 2015

@author: FranciscoP.Romero

Modified on Wed Sep 27 17:20:12 2017 by:    
    GONZALO PEREZ FERNANDEZ.
    GREGORIO BALDOMERO PATINO ESTEO.
    SERGIO FERNANDEZ GARCIA.
"""

import codecs
from numpy import corrcoef, transpose, arange
from pylab import show, colorbar, xticks, yticks
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 0. Load Data
f1 = codecs.open("../data/dengue_pivot_select.csv", "r", "utf-8")
cities = []
read_header = False
for line in f1:
	if read_header: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
        
		if row != []:
			cities.append(map(float, row))
	read_header = True

f2 = codecs.open("../data/dengue_pivot_mean.csv", "r", "utf-8")
cities_mean = []
read_header = False
for line in f2:
	if read_header: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
        
		if row != []:
			cities_mean.append(map(float, row))
	read_header = True

# plotting the correlation matrix
#http://glowingpython.blogspot.com.es/2012/10/visualizing-correlation-matrices.html

# The total number of characteristics is the same in both
features_length =len(cities[0])

R1 = corrcoef(transpose(cities))
plt.pcolormesh(R1)
#pcolor(R1)
colorbar()
yticks_r1 = yticks(arange(0,features_length),range(0,features_length))
xticks_r1 = xticks(arange(0,features_length),range(0,features_length))
plt.title('Correlation suppresing empty data')
show()

R2 = corrcoef(transpose(cities_mean))
#pcolor(R2)
plt.pcolormesh(R2)
colorbar()
yticks_r2 = yticks(arange(0,features_length),range(0,features_length))
xticks_r2 = xticks(arange(0,features_length),range(0,features_length))
plt.title('Correlation with arithmetic means')
show()

# http://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
# Generate a mask for the upper triangle
sns.set(style = 'white')
mask = np.zeros_like(R1, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(R1, mask=mask, cmap=cmap, vmax=.8,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_title('Correlation suppresing empty data')

mask = np.zeros_like(R2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(R2, mask=mask, cmap=cmap, vmax=.8,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_title('Correlation with arithmetic means')