#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:45:27 2017

@authors:   
    GONZALO PEREZ FERNANDEZ.
    GREGORIO BALDOMERO PATINO ESTEO.
    SERGIO FERNANDEZ GARCIA.
            
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import preprocessing 
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz 
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score

def load_data(file, filter_parameters = None, excludes_features = None,
              outliers = None):
    
    df = pd.read_csv(file)
    
    df.replace('', np.nan, inplace=True, regex=True)
  
    if filter_parameters != None:
        for key, value in filter_parameters.iteritems():
            df_aux = df.loc[df[key].isin(value)]
            df=df_aux
    
    if excludes_features != None:
        if type(excludes_features)==list and type(excludes_features[0])== str:
            df.drop(labels = excludes_features, axis = 1, inplace = True)
        elif type(excludes_features)==list and type(excludes_features[0])== int:
            df.drop(df.columns[excludes_features], axis = 1, inplace = True)
            
    if outliers != None:
        df.drop(outliers, axis = 0, inplace = True)
    
    return df


def normalization_with_minmax(data):
    
    min_max_scaler = preprocessing.MinMaxScaler()
        
    return min_max_scaler.fit_transform(data.dropna().values)  


def main():
    # 0. Load data
    years = range(2004, 2011)
    features_excluded = ['week_start_date']
    _filter = {'city':['sj'], 'year':years}
    _outliers = [732,769]
    
    data_train = load_data(
           "../data/dengue_features_train.csv",
            filter_parameters = _filter, excludes_features = features_excluded,
              outliers = _outliers)

    
    data_labels = load_data(
            "../data/dengue_labels_train.csv",
            filter_parameters = _filter)
    
    mergedf = pd.merge(data_train, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')
    plt.rcdefaults()
    mergedf.hist(figsize = (20,20))
    plt.show()
    
    mergedf.plot(kind = 'density', subplots = True, layout=(12, 2), sharex = False, figsize = (20,20))
    plt.show()
    
    scatter_matrix(mergedf, figsize=(80, 80), diagonal='kde')
    plt.show()
    
    #Remove no useful features in standardization
    mergedf_drop = mergedf.drop(['city', 'year', 'weekofyear'], axis = 1, inplace = False)
    features= mergedf_drop.columns.tolist()[:-1]
    
    norm_df = normalization_with_minmax(mergedf_drop)
    
    # 1. Correlation
    corr = map(lambda x1: pearsonr(x1,norm_df[-1].tolist())[0],
               [norm_df[x].tolist() for x in range(len(features))])
    
    y_pos = np.arange(len(features))
    
    plt.bar(y_pos, corr, align='center')
    plt.xticks(y_pos, features, rotation = 90)
    plt.ylabel('Correlation')
    plt.title('Correlation features vs target')
    plt.show()

    features_corr = zip(features, corr)
    print tabulate(features_corr, headers = ['Feature','R value'])
    
    #Selection of characteristics with correlation greater than 0.7.
    features_selected = [features_corr[i][0] for i in range(len(features_corr))
                            if abs(features_corr[i][1]) > 0.7]
    
    # 2. CROSS VALIDATION ANALYSIS
    df_dropna = mergedf_drop.dropna()
    total_scores = []

    for i in range(2, 30):
        regressor = DecisionTreeRegressor(max_depth=i)
        regressor.fit(df_dropna[features_selected], df_dropna['total_cases'])
        scores = -cross_val_score(regressor, df_dropna[features_selected],
                 df_dropna['total_cases'], scoring='neg_mean_absolute_error', cv=10)
        total_scores.append(scores.mean())
    
    plt.plot(range(2,30), total_scores, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('cv score')
    plt.show()
    
    best_max_depth_mean = total_scores.index(min(total_scores))+2
    print 'Best MAX_DEPTH: %d' % (best_max_depth_mean)
    
    #3. Build the model
    
    #3.1 Model Parametrization 
    # criterion: mse mean squared error, which is equal to variance reduction as feature selection criterion
    # splitter: best/random
    # max_depth: low value avoid overfitting
    
    regressor = DecisionTreeRegressor(criterion = 'mse', max_depth = best_max_depth_mean,
                                      random_state = 0)

    #3.2 Model construction
    regressor.fit(df_dropna[features_selected], df_dropna['total_cases'])
    
    
    #3.3.  Model Visualization
    dot_data = export_graphviz(regressor, out_file='tree.dot', 
                               feature_names = features_selected,
                               filled=True, rounded=True,  
                               special_characters=True)  
    
    graph = graphviz.Source(dot_data, format = 'png')  
    dot_data = export_graphviz(regressor, out_file='tree.dot') 
    graph = graphviz.Source(dot_data , format = 'png') 
    graph.render("dengue_cases") 
    graph 
    """
    dot_data = export_graphviz(regressor, out_file='tree.dot',
                               feature_names = features_selected, 
                               filled=True, rounded=True)
    graph = graphviz.Source(dot_data)  
    graph.render(dot_data, view=True)
    graph
    """
    # 3.4 Feature Relevances
    
    print '\n\t\t[ RELEVANCES FEATURES ]\n'
    relevances_list = zip(features_selected, regressor.feature_importances_)
    
    print tabulate(relevances_list, headers=['Feature selected', 'Relevance'])
    
    reg_values = list(regressor.feature_importances_)
    reg_values_sort = sorted(reg_values, reverse = True)
    
    max_values_feature = []
    
    for i in range(2):
        pos = reg_values.index(reg_values_sort[i])
        max_values_feature.append(features_selected[pos])
    
    print '\nFeatures with more relevance: \n\t' + str(max_values_feature)
    
    
    #-----
    # On the other hand, we compare with other metric.
    # Compute the max 
    mae = []

    for i in range(2, 30):
        regressor = DecisionTreeRegressor(max_depth=i)
        regressor.fit(df_dropna[features_selected], df_dropna['total_cases'])
        pred_values = regressor.predict(df_dropna[features_selected])
        maev = mean_absolute_error(df_dropna['total_cases'],pred_values)
        mae.append(maev)
        
    # Plot mae   
    plt.plot(range(2,30), mae, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('mae')
    plt.show()
    #-----
    
    best_max_depth_mae = mae.index(min(mae))+2
    print 'Best MAX_DEPTH with MAE: %d' % (best_max_depth_mae)
    
    # 2.2 Feature Relevances with MAE
    
    print '\n\t\t[ RELEVANCES FEATURES(MAE)]\n'
    relevances_list = zip(features_selected, regressor.feature_importances_)
    
    print tabulate(relevances_list, headers=['Feature selected', 'Relevance'])
    
    reg_values = list(regressor.feature_importances_)
    reg_values_sort = sorted(reg_values, reverse = True)
    
    max_values_feature = []
    
    for i in range(2):
        pos = reg_values.index(reg_values_sort[i])
        max_values_feature.append(features_selected[pos])
    
    print '\nFeatures with more relevance: \n\t' + str(max_values_feature)

    
if __name__ == '__main__':
	main()