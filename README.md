# MACHINE-LEARNING-TECHNIQUES

## Authors:
    - GONZALO PEREZ FERNANDEZ.
    - GREGORIO BALDOMERO PATINO ESTEO.
    - SERGIO FERNANDEZ GARCIA.

This is the repository of the team MLT_ESI from the drivendata.org competition about Dengue virus (https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/). The goal is to predict the total cases for each city, year and week of the year in the test set. To reach that goal we have follow some steps, dividing the main goal into little goals to make it easier and get a better approach of the prediction. The steps that we follow are that:

TASK_1
First, we selected a sample from the whole data. We take the data from San Juan since 2004 to 2010. When we extracted that sample, a PCA was executed and the result was plotted. Some conclusion was extracted from the results in the way that we can extract some knowledge from that conclusion.

TASK_2
The main goal of this task is to make hierarchical clustering. To reach that goal the first thing we was normalize the dataset to improve the precision when we extract the correlations. Then we compute a similarity matrix to execute a hierarchical clustering algorithm.

TASK_3
The goal of this task is to apply K-Means algorithm to our data.

TASK_4
The goal of this task is to obtain the best max_depth parameter from our data. To reach that we have studied the correlation between features and total cases. Then we select a subset of features using the knowledge we extract from the last tasks. After that we should build the Decision Tree and we also obtain the features relevancies. After that we must perform a cross validation process in order to reach the goal of the task
