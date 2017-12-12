#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:01:43 2017

@author: sergio
"""
import pandas as pd

def write_result(col_names, submission_data, prediction_knn, prediction_rf, prediction_path, name_file, first):
    
    name_pred_1 = '_sj_knn_iq_knn_'
    name_pred_2 = '_sj_knn_iq_rf_'
    name_pred_3 = '_sj_rf_iq_knn_'
    name_pred_4 = '_sj_rf_iq_rf_'
    
    final_data_knn=[]
    final_data_rf=[]
        
    for i in range(len(submission_data)):
        row_knn = []
        row_rf = []
        
        
        submission_city = submission_data.iloc[i]['city']
        submission_year = submission_data.iloc[i]['year']
        submission_weekofyear = submission_data.iloc[i]['weekofyear']
        
        row_knn.append(submission_city)
        row_rf.append(submission_city)
        
        row_knn.append(submission_year)
        row_rf.append(submission_year)
        
        row_knn.append(submission_weekofyear)
        row_rf.append(submission_weekofyear)
        
        row_knn.append(int(prediction_knn[i]))
        row_rf.append(int(prediction_rf[i]))
        
        final_data_knn.append(row_knn)
        final_data_rf.append(row_rf)
        
    df_knn = pd.DataFrame(final_data_knn, columns = col_names)
    df_rf = pd.DataFrame(final_data_rf, columns = col_names)
            
    
    if first:
        df_knn.to_csv(prediction_path + '/predictions' + name_pred_1 +
                      name_file + '.csv', index=False, sep=',', encoding='utf-8')    
        df_knn.to_csv(prediction_path + '/predictions' + name_pred_2 +
                      name_file + '.csv', index=False, sep=',', encoding='utf-8')    
        df_rf.to_csv(prediction_path + '/predictions' + name_pred_3 +
                      name_file + '.csv', index=False, sep=',', encoding='utf-8')    
        df_rf.to_csv(prediction_path + '/predictions' + name_pred_4 +
                      name_file + '.csv', index=False, sep=',', encoding='utf-8')            
        first = False
    
    else:
        with open(prediction_path + '/predictions' + name_pred_1 + name_file +'.csv', mode='a') as _file:
            df_knn.to_csv(_file, header=False, index=False, sep=',', encoding='utf-8')
        
        with open(prediction_path + '/predictions' + name_pred_3 + name_file +'.csv', mode='a') as _file:
            df_knn.to_csv(_file, header=False, index=False, sep=',', encoding='utf-8')

        with open(prediction_path + '/predictions' + name_pred_2 + name_file +'.csv', mode='a') as _file:
                df_rf.to_csv(_file, header=False, index=False, sep=',', encoding='utf-8')
        
        with open(prediction_path + '/predictions' + name_pred_4 + name_file +'.csv', mode='a') as _file:
            df_rf.to_csv(_file, header=False, index=False, sep=',', encoding='utf-8') 
                