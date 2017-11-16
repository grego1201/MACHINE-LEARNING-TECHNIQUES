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
        data_for_cluster = data.drop(labels = ['city', 'year'], axis = 1, inplace = False)        
        data_test_hiech = data_for_cluster
        
        # Outliers will be deleted
        elements, outliers = clustering.hierarchical_clustering(data_test_hiech)
        print '\nOutliers in: ' + city   
        while (outliers != None):
            print 'Auto-detected Outliers: \n\t' + str(outliers)
            outliers_list = outliers[0]
            for i in range(1, len(outliers)):
                outliers_list += outliers[i]
            
            data_test_hiech.drop(outliers_list, axis = 0, inplace = True)
            elements, outliers = clustering.hierarchical_clustering(data_test_hiech)
        
        
        # Join data
        merge_data = pd.merge(data, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')
        merge_data.drop(labels = ['city', 'year'], axis = 1, inplace = True)
        merge_data.dropna(inplace=True)
        
        # Croos Validation for select features
        feature_selected, max_deph = cros.cross_validation(merge_data, verbose = False)
        
        # Regressor for select relevant features
        relevant_features = reg.tree_regressor(merge_data, max_deph, feature_selected, 'total_cases', city, False)
        
        all_revelant_features[city] = relevant_features
    
    print '\n\t [ SELECTED FEATURES ]'
    for key, value in all_revelant_features.iteritems():
        print 'City: %s, %2d features: \n\t %s' % (key, len(value), str(value))
    
if __name__ == '__main__':
	main()