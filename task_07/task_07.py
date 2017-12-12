#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:32:45 2017

@author: sergio
"""
import datetime
from loaddata import load_data, get_values_of
from writer import write_result
import crossvalidation as cros
import regressor as reg
import clustering
import prediction as predict
import pandas as pd
import os

def count_elements(elements):
    total = 0
    if type(elements)==list:
        total=len(elements)
        
    elif type(elements)==dict:
        for key, value in elements.iteritems():
            total += len(value)
    
    return total

def assign_name():
    x = datetime.datetime.now()
    
    
    month = str(x.month) if x.month > 9 else ('0' + str(x.month))
    day = str(x.day) if x.day > 9 else ('0' + str(x.day))
    hour = str(x.hour) if x.hour > 9 else ('0' + str(x.hour))
    minute = str(x.minute) if x.minute > 9 else ('0' + str(x.minute))
    second = str(x.second) if x.second > 9 else ('0' + str(x.second))
        
    name = str(x.year) + month + day + '_' + hour + minute + second
    
    return name

def data_fill_mode(data, mode):
    
    if mode != None:
        if type(mode)==str:
            if mode == 'dropna':
                data.dropna(how = 'any', inplace = True)
            if mode == 'interpolate':
                data.interpolate(method='linear', inplace = True)
            if mode == 'mean':
                data.fillna(data.mean(), inplace = True)
            
        if type(mode)==list:
            data.interpolate(method='linear', limit = 3, inplace = True)
            
            if mode[1] == 'dropna':
                data.dropna(how = 'any', inplace = True)
            if mode[1] == 'mean':
                data.fillna(data.mean(), inplace = True)
                
        return data
 

    
    
def main():

    first = True
    name_file = assign_name()
    prediction_path = '../predictions/'+ name_file
    
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
        
    
    years = None
    features_excluded = ['week_start_date']

    _outliers = None

    cities = get_values_of("../data/dengue_features_train.csv", 'city')
    
    target = 'total_cases'
    
    all_revelant_features = {}
    all_scores = []
    
    modes = [#'dropna', 'interpolate', 'mean',
                 ['interpolate', 'mean']]#, ['interpolate', 'dropna']]

    for mode in modes:

        first = True
        scores_city = {}
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
            

            
            data_fill = data_fill_mode(data, mode)
            data_labels_fill = data_fill_mode(data_labels, mode)            
        
        
        
        
            # Adapt data for clustering
            data_test_hiech = data_fill.drop(labels = ['city', 'year'],
                                             axis = 1, inplace = False)        
            
            # Outliers will be deleted
            elements, outliers, cut = clustering.hierarchical_clustering(
                    data = data_test_hiech, verbose = False)
        
            n_element = count_elements(elements)
            n_outliers = count_elements(outliers)        
            total = n_element + n_outliers
            
            print 'Analysis in: %s on mode %s' % (city, str(mode))
        
            total_outliers = []
            while (outliers != None):  
                total_outliers += outliers
                data_test_hiech.drop(outliers, axis = 0, inplace = True)
                elements, outliers, cut = clustering.hierarchical_clustering(
                        data_test_hiech, cut = cut, 
                        first_total = total, 
                        verbose = False)
            
            if total_outliers:
                print 'Auto-detected Outliers:'
                print total_outliers
        
        
            # Join data
            data_without_outliers = data_fill
            data_without_outliers.drop(total_outliers, axis = 0, inplace = True)
            
            merge_data = pd.merge(data_without_outliers, data_labels_fill,
                                  on = ['city', 'year', 'weekofyear'],
                                  how = 'inner')
            first_year = merge_data['year'].min()
            last_year = merge_data['year'].max()
            split_year = int(last_year - round((last_year -first_year)*0.2))
                        
            
            
            
            # Features clustering
            data_for_features = merge_data.drop(labels = ['city', 'total_cases'], axis = 1)
            
            feature_groups = clustering.hierarchical_clustering_features(data_for_features, verbose = False)
            
            # Croos Validation for select features
            features_selected, max_deph = cros.cross_validation(merge_data, feature_groups, split_year, 
                                            target = target)
            
            # Regressor for select relevant features
            relevant_features = reg.tree_regressor(merge_data, split_year, 
                                                   max_deph,
                                                   features_selected, 
                                                   target, city, 
                                                   verbose = False)
            
            
            
            all_revelant_features[city] = relevant_features
            
            all_features = merge_data.columns.tolist()[1:-1]
            

            
            data_Test = load_data(
                   "../data/dengue_features_test.csv",
                    filter_parameters = _filter, 
                    excludes_features = features_excluded,
                    outliers = _outliers)

            # prediction        
        
            prediction_knn, score_knn = predict.knn_prediction(merge_data, split_year,
                                    features_selected, target,
                                    data_Test, verbose = True)
            print('Score KNN on %s mode is : %.4f' % (mode, score_knn))
            
            prediction_rf, score_rf = predict.rf_prediction(merge_data, split_year,
                                    all_features, target,
                                    data_Test, verbose = True)
            print('Score RandomForest on %s mode is : %.4f' % (mode, score_rf))
        
            
            scores_city[city] = [(mode,'Knn', score_knn),(mode,'RF', score_rf)] 
            
            # Load submission data file.
            submission_data = load_data("../data/submission_format.csv",
                                         filter_parameters = _filter)
        
        
            # wr ite the results in a csv file
            # Write result files.
            col =["city","year","weekofyear","total_cases"]
            write_result(col, submission_data, prediction_knn, prediction_rf, 
                         prediction_path, (name_file + str(mode)) , first)
            first = False
            
        all_scores.append(scores_city)

    print all_scores

    """ 
    print '\n\t [SCORES]'
    
    for key, value in all_scores.iteritems():
        print 'City: %s, %2d features: \n\t %s' % (key, len(value), str(value))
    """
    
if __name__ == '__main__':
	main()