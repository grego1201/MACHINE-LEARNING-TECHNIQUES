# MACHINE-LEARNING-TECHNIQUES

## Authors:
    - GONZALO PEREZ FERNANDEZ.
    - GREGORIO BALDOMERO PATINO ESTEO.
    - SERGIO FERNANDEZ GARCIA.

This is the repository of the team MLT_ESI from the drivendata.org competition about Dengue virus (https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/). The goal is to predict the total cases for each city, year and week of the year in the test set. To reach that goal we have followed some steps, dividing the main goal into little goals to make it easier and get a better approach of the prediction. The steps which we follow are this:

### TASK_1

Firstly, we selected a sample from the whole data. We took the data from San Juan between 2004 and 2010. When we extracted that sample, a PCA was executed and the result was plotted. Some conclusion was extracted from the results in the way that we can extract some knowledge from that conclusion.

### TASK_2

The main goal of this task is to make hierarchical clustering. To reach that goal, the first thing was to normalize the dataset in order to improve the precision when we extracted the correlations. Then, we computed a similarity matrix to execute a hierarchical clustering algorithm.

### TASK_3

The goal of this task is to apply K-Means algorithm to our data.

### TASK_4

The goal of this task is to obtain the best max_depth parameter from our data. To reach it, we have studied the correlation between features and total cases. Then, we selected a subset of features using the knowledge which we extracted from the last tasks. After it, we should build the Decision Tree and also, we obtain the features relevancies. After it, we must perform a cross validation process in order to reach the goal of the task.

### TASK_5

The task goal is to obtain the most relevants features of each city with all data, with the files 'dengue_training_features' and 'dengue_training_labels'. For this task we are going to use the learned methods.

### TASK_6

Task goal, obtain predictions using the file 'dengue_features_train' and 'dengue_labels_train'. Firstly we are going to select these features which are most importanto to this task. For this, we are goint to do this for 'sj' San Juan and 'iq' iquitos. And with this training we will execute the KNN algorithm to obtain predictions.

### TASK_7

Final resilts.
