#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:24:19 2017

@author: sergio
"""
import numpy as np
import pandas as pd


def load_data(_file, filter_parameters = None, excludes_features = None,
              outliers = None):
    
    df = pd.read_csv(_file)
    
    df.replace('', np.nan, inplace=True, regex=True)
  
    if filter_parameters != None:
        for key, value in filter_parameters.iteritems():
            if (value!= None):
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

def get_values_of(_file, feature):
    df = load_data(_file)
    a = list(set(df[feature]))

    return a
    