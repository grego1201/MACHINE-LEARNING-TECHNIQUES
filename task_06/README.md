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

![sj_distancia][3] ![sj_uniforme][4]

* __Iquitos:__

![iquitos_distancia][5] ![iquitos_uniforme][6]

### Prediction
After all, we will charge the test file 'dengue_features_test' and we will obtain the prediction for this data with the neighbours and the distance parameter in order to predict the total cases.

* __San Juan:__
![SanJuan_cv][7] ![SanJuan_training][8]

* __Iquitos:__
![Iquitos_cv][9] ![Iquitos_training][10]

### Score.

![Score][11]



[1]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Cross-validation/sj_cross.png
[2]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Cross-validation/iquitos_cross.png
[3]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Build-model/sj_distancia.png
[4]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Build-model/sj_uniform.png
[5]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Build-model/iquitos_distancia.png
[6]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Build-model/iquitos_uniform.png
[7]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/SanJuan_cv.png
[8]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/SanJuan_training.png
[9]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Iquitos_cv.png
[10]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/Iquitos_training.png
[11]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_06/images/scores.png
