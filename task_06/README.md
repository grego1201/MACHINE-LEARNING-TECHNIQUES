# Task 06
    
## Task Goal.
The task goal is to obtain predictions using the file 'dengue_features_train' and 'dengue_labels_train'. Firstly we are going to select these features which are most importanto to this task. For this, we are goint to do this for 'sj' San Juan and 'iq' iquitos. And with this training we will execute the KNN algorithm to obtain predictions.

## Features Selection.
First to all, we are going to use the data obtain in the previous task, which are the most important features to train the data.


### Load Data.
As we mentioned before, we are going to obtain the training data from 'dengue_features_train' and 'dengue_labels_train. for each city and we will parameterize this data. We will remove the outliers with an automatic method.

### Cross validation Method
To realize the cross validation we put together the data without outliers and the data labels considering the features 'city', 'year' and 'weekofyear' and with this method we will obtain the value of the feature 'total_cases'.

Now, with the most relevants features, total_cases and the data without 'city'and 'year' we we will obtain the neighbours.

* __San Juan:__

![sj_cross][1]


* __Iquitos:__

![iquitos_cross][2]

### Build model

When we have obtain the neighbour number we are going to buil de KNN model using 'uniform' and 'distance' as parameters.

* __San Juan:__

![sj_training][3] 

* __Iquitos:__

![iquitos_training][4]


### Prediction
After all, we will charge the test file 'dengue_features_test' and we will obtain the prediction for this data with the neighbours and the distance parameter in order to predict the total cases.

* __San Juan:__

![SanJuan_prediction][5]

* __Iquitos:__

![Iquitos_prediction][6]


### Score.

![Score][11]



[1]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/SanJuan_cv.png
[2]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Iquitos_cv.png
[3]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/SanJuan_training.png
[4]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Iquitos_training.png
[5]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/SanJuan_prediction.png
[6]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Iquitos_prediction.png
[7]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/scores.png
