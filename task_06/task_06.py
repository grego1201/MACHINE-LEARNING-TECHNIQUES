#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:32:45 2017

@author: sergio
"""
from loaddata import load_data, get_values_of
import crossvalidation as cros
import regressor as reg
import clustering
import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
import csv
import codecs
from loaddata_entrenamiento import load_data_entrenamiento

def count_elements(elements):
    total = 0
    if type(elements)==list:
        total=len(elements)
        
    elif type(elements)==dict:
        for key, value in elements.iteritems():
            total += len(value)
    
    return total

def main():
       
    years = None
    features_excluded = ['week_start_date']

    _outliers = None

    cities = get_values_of("../data/dengue_features_train.csv", 'city')

    all_revelant_features = {}
    for city in cities:
        # Filtering by values of the keys
        _filter = {'city':[city], 'year':years}
        
        #Load city data
        data = load_data(
               "../data/dengue_features_train.csv",
                filter_parameters = _filter, excludes_features = features_excluded,
                  outliers = _outliers)
        
        # Load total cases by city, year and week of year
        data_labels = load_data("../data/dengue_labels_train.csv",filter_parameters = _filter)    
        
        
        # Adapt data for clustering
        data_test_hiech = data.drop(labels = ['city', 'year'], axis = 1, inplace = False)        
        
        # Outliers will be deleted
        elements, outliers, cut = clustering.hierarchical_clustering(data = data_test_hiech)
    
        n_element= count_elements(elements)
        n_outliers = count_elements(outliers)        
        total=n_element + n_outliers
        
        print 'Analysis in: %s' % (city)
       


        total_outliers = []
        while (outliers != None):  
            total_outliers += outliers
            data_test_hiech.drop(outliers, axis = 0, inplace = True)
            elements, outliers, cut = clustering.hierarchical_clustering(data_test_hiech,
                                                                cut = cut,
                                                                first_total = total)
        
        if total_outliers:
            print 'Auto-detected Outliers:'
            print total_outliers
        
        # Join data
        data_without_outliers = data
        data_without_outliers.drop(total_outliers, axis = 0, inplace = True)
        
        merge_data = pd.merge(data_without_outliers, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')
        merge_data.drop(labels = ['city', 'year'], axis = 1, inplace = True)
        merge_data.dropna(inplace = True)
        
        # Features clustering
        data_for_features = merge_data.drop(labels = ['total_cases'], axis = 1)
        clustering.hierarchical_clustering_features(data_for_features)
        
        # Croos Validation for select features
        feature_selected, max_deph = cros.cross_validation(merge_data, algorithm = 'DecisionTreeRegressor')
        
        # Regressor for select relevant features
        relevant_features = reg.tree_regressor(merge_data, max_deph, feature_selected, 'total_cases', city)
        
        all_revelant_features[city] = relevant_features
        
        # For each city, one model KNN
        # Croos Validation for select features
        target = ['total_cases']
        n_neighbors, X, y = cros.cross_validation(merge_data, algorithm ='KNN', 
                              features = relevant_features, 
                              target = target, verbose = False)
        
        #---------------------------------------------

        # prediction
        
        dataTest = data_without_outliers.dropna(inplace = False)
        
        test = dataTest[relevant_features]
        
        
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
        prediccion = knn.fit(X,y).predict(test)
        
        
        # show prediction
        print "\nPREDICTION:"
        xx = np.stack(i for i in range(len(prediccion)))
        plt.plot(xx, prediccion, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,'distance'))
        
        plt.show()
        
        # write the results in a csv file
        data_rellenar, names = load_data_entrenamiento("../data/submission_format.csv")
        for i in range(len(data_rellenar)):
            data_rellenar[i][3]=int(prediccion[i][0])
        
        col =["city","year","weekofyear","total_cases"]
        df = pd.DataFrame(data_rellenar, columns = col)
        df.to_csv("../data/datos_entrenamiento.csv", index=False, sep=',', encoding='utf-8')    
    

        #---------------------------------------------
        
        
        
    print '\n\t [ SELECTED FEATURES ]'
    for key, value in all_revelant_features.iteritems():
        print 'City: %s, %2d features: \n\t %s' % (key, len(value), str(value))
    
    
    
    
    
if __name__ == '__main__':
	main()