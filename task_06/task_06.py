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
    target = ['total_cases']
    
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
        
        merge_data = pd.merge(data_without_outliers, data_labels,
                              on = ['city', 'year', 'weekofyear'], how = 'outer')
        merge_data.drop(labels = ['city', 'year'], axis = 1, inplace = True)
        merge_data.dropna(inplace = True)
        
        # Features clustering
        data_for_features = merge_data.drop(labels = target, axis = 1)
        clustering.hierarchical_clustering_features(data_for_features)
        
        # Croos Validation for select features
        feature_selected, max_deph = cros.cross_validation(merge_data,
                                        algorithm = 'DecisionTreeRegressor')
        
        # Regressor for select relevant features
        relevant_features = reg.tree_regressor(merge_data, max_deph,
                                               feature_selected, target, city)
        
        all_revelant_features[city] = relevant_features
        
        # For each city, one model KNN
        # Croos Validation for select features
        n_neighbors, X, y = cros.cross_validation(merge_data, algorithm ='KNN', 
                              features = relevant_features, 
                              target = target, verbose = True)
        
        #---------------------------------------------

        # prediction
        data_Test = load_data(
               "../data/dengue_features_test.csv",
                filter_parameters = _filter, excludes_features = features_excluded,
                  outliers = _outliers)
        
        #data_Test.dropna(inplace = True)
        test = data_Test[relevant_features]
        test.interpolate(method='linear', inplace = True)
        
        
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
        prediction = knn.fit(X,y).predict(test)
        
        
        # show prediction
        print "\nPREDICTION:"
        xx = np.stack(i for i in range(len(prediction)))
        plt.plot(xx, prediction, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,'distance'))
        
        plt.show()
        
        # write the results in a csv file
        submission_data = load_data("../data/submission_format.csv",
                                         filter_parameters = _filter)
        final_data=[]
        
        for i in range(len(final_data)):
            row=[]
            
            row.append(submission_data.iloc[i]['city'])
            row.append(submission_data.iloc[i]['year'])
            row.append(submission_data.iloc[i]['weekofyear'])
            row.append(int(prediction[i]))
            
            final_data.append(row)
        
        col =["city","year","weekofyear","total_cases"]
        df = pd.DataFrame(final_data, columns = col)
        df.to_csv('../data/predictions_for_'+ city +'.csv', index=False, sep=',', encoding='utf-8')    
        
        #---------------------------------------------
        
    print '\n\t [ SELECTED FEATURES ]'
    for key, value in all_revelant_features.iteritems():
        print 'City: %s, %2d features: \n\t %s' % (key, len(value), str(value))
    
    
if __name__ == '__main__':
	main()