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
    #Load data from San Juan
    years = None
    features_excluded = ['week_start_date']

    _outliers = None

    cities = get_values_of("../data/dengue_features_train.csv", 'city')
    
    for city in cities:
        _filter = {'city':[city], 'year':years}
        dataSanJuan = load_data(
               "../data/dengue_features_train.csv",
                filter_parameters = _filter, excludes_features = features_excluded,
                  outliers = _outliers)
        
        
        #Load data from Iquito
        #_filter = {'city':['iq'], 'year':years}
        #    dataIquitos = load_data(
        #           "../data/dengue_features_train.csv",
        #            filter_parameters = _filter, excludes_features = features_excluded,
        #            outliers = _outliers)
        
        data_labels = load_data(
                "../data/dengue_labels_train.csv",
                filter_parameters = _filter)
        
        
        
        dataSanJuan_for_cluster = dataSanJuan.drop(labels = ['city', 'year'], axis = 1, inplace = False)        
        data_test_Sanjuan_hiech = dataSanJuan_for_cluster 
        data_test_Sanjuan_km = dataSanJuan_for_cluster
        
        
        elements, outliers = clustering.hierarchical_clustering(data_test_Sanjuan_hiech)
           
        while (outliers != None):
            print 'Auto-detected Outliers: \n\t' + str(outliers)
            outliers_list = outliers[0]
            for i in range(1, len(outliers)):
                outliers_list += outliers[i]
            
            data_test_Sanjuan_hiech.drop(outliers_list, axis = 0, inplace = True)
            elements, outliers = clustering.hierarchical_clustering(data_test_Sanjuan_hiech)
         
        print '\n\tAuto-deleted Outliers'
        
        
        clustering.hierarchical_clustering_features(dataSanJuan_for_cluster)
        
        
        elements_km, outliers_km = clustering.kMeans_clustering(data_test_Sanjuan_km)
        
        while (outliers_km != None):
            print 'Auto-detected Outliers: \n\t' + str(outliers_km)
            outliers_list = outliers_km[0]
            for i in range(1, len(outliers_km)):
                outliers_list += outliers_km[i]
            
            data_test_Sanjuan_km.drop(outliers_list, axis = 0, inplace = True)
            elements_km, outliers_km = clustering.kMeans_clustering(data_test_Sanjuan_km)
         
        print '\n\tAuto-deleted Outliers'    
        
        
           
        
        mergedf_SanJuan = pd.merge(dataSanJuan, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')
        mergedf_SanJuan.drop(labels = ['city', 'year'], axis = 1, inplace = True)
        
        mergedf_SanJuan.dropna(inplace=True)
        feature_selected, max_deph = cros.cross_validation(mergedf_SanJuan)
        reg.tree_regressor(mergedf_SanJuan, max_deph, feature_selected, 'total_cases', city)
 
if __name__ == '__main__':
	main()