# Task 05
    
## Task Goal

The task goal is to obtain the most relevants features of each city with all data, with the files 'dengue_training_features' and 'dengue_training_labels'.
For this task we are going to use the learn methods.

## Load Data

Firstly, we load the data, but we are goint to do this twice because we have two cities to analyse. In addition, in this case we just are going to exclude 'week_start_date' and later we are going to remove the year.
Except it, we do it with the same mode than the before tasks. 

## Features selection

We detecting the outliers to remove it and to better analyse the data.

To detect it, we make a hierarchical clustering whit the 'single' method because it is the best outliers detector and the best distance method.
we will make the cut by 3.7 due to both cities separate it enough to remove the most prominent outliers. 

Furthermore, we add a way to identify if the smaller group is so big to do it smaller too, and don't remove so much elements which don't be outliers.
To do the groups we will check data using kmeans.
When we have eliminated the outliers and with the groups created, we are going to use one feature of each group, it will say the samebecause it is related and due to we don't do more noise than necesary.
To select the important features we will do a "cross validation" with a decision tree with which we are going to know some relevants features.

## Conclusion

The most relevants features to Iquitos are:
    ['reanalysis_specific_humidity_g_per_kg', 'weekofyear']
And to San Juan are: 
    ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_avg_temp_k']