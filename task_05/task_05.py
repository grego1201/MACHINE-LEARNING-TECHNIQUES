#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:47:35 2017

@author: sergio
"""
from loaddata import load_data, get_values_of
import crossvalidation as cros
import regressor as reg
import clustering
import pandas as pd


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
        elements, outliers, cut = clustering.hierarchical_clustering(data = data_test_hiech,
                                                            verbose = True)
        n_element= count_elements(elements)
        n_outliers = count_elements(outliers)        
        total=n_element + n_outliers
        
        print '\nOutliers in: %s \n\t' % (city)
        


        total_outliers = []
        while (outliers != None):  
            total_outliers += outliers
            data_test_hiech.drop(outliers, axis = 0, inplace = True)
            elements, outliers, cut = clustering.hierarchical_clustering(data_test_hiech,
                                                                cut = cut,
                                                                first_total = total, 
                                                                verbose = True)
        
        if total_outliers:
            print 'Auto-detected Outliers:'
            print total_outliers
        
        # Join data
        data_without_outliers = data
        data_without_outliers.drop(total_outliers, axis = 0, inplace = True)
        
        merge_data = pd.merge(data_without_outliers, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')
        merge_data.drop(labels = ['city', 'year'], axis = 1, inplace = True)
        merge_data.dropna(inplace = True)
        
        clustering.hierarchical_clustering_features(merge_data, verbose = True)
        
        # Croos Validation for select features
        feature_selected, max_deph = cros.cross_validation(merge_data, verbose = True)
        
        # Regressor for select relevant features
        relevant_features = reg.tree_regressor(merge_data, max_deph, feature_selected, 'total_cases', city, verbose = True)
        
        all_revelant_features[city] = relevant_features
    
    print '\n\t [ SELECTED FEATURES ]'
    for key, value in all_revelant_features.iteritems():
        print 'City: %s, %2d features: \n\t %s' % (key, len(value), str(value))
    
if __name__ == '__main__':
	main()