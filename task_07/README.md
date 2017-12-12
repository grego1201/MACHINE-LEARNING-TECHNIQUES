# Task 07
    
## 1. The approach to obtain the firsts results in Driven Data
To approach the correct analysis of the main decisions that we have led us to a first result in the prediction, we must take into account those that mainly affect the treatment of the data and also the meanings that we give them to make decisions.

### 1.1 Treatment of the data
In the data provided to carry out the study, “dengue features train.csv”, there are samples with features without value, at first moment when features without data are removed that copy (record or entry) This decision extends to clustering, normalization data with MinMAx. Since it is not possible to delete samples in the “dengue features test.csv” and “submission format.csv” files, interpolation is made in the missing data in the features of each one of the samples.

### 1.2 Features selection
For features selection we rely on the correlation values with the features used to predict ’total cases’, if they have correlation value greater than 0.7 these characteristics are taken and used to a DecisionTreeRegresor that will give us the most relevant final featured to be used in the KNeighborsRegressor algorithm.

**The first value was a SCORE of 26.8462.**

## 2. Main lines of improvement applied.
1. First improve option was include RandomForestRegressor algorithm. At the beginning it just gives a little improve with Iquitos, meanwhile it was worst with San Juan.
2. At the same time some test has done with different strategies when there is missing data. Those options were contemplated:

    a) Drop rows with any missing value.

    b) Fill with mean of feature the missing value.

    c) Fill with interpolate method the missing value.

    d) Fill with interpolate method the missing value with as max 3 consecutive values and if still is null, drop the row.

    e) Fill with interpolate method the missing value with as max 3 consecutive values and if still is null, fill it with mean.

   The needed data of dengue features test.csv and submission format.csv files, if the strategy is different to media and/or interpolate, the default option complete interpolate.

3. In other hand, when the predictions were done, training and test data were separated. To divide them the years from bigger to lowest value was taken, then next formula.

```
    split year = int(last year - round((last year -first year)*0.2))
```

  *split_year* is the year were the division will be. Less than split year will be use for training and the rest for test.

4. To select the relevant features to use in KneighborsRegressor we make a features cluster. Then we divided it in groups, taken the feature with highest correlation in the group.
5. Some code was changed in CrossValidation with a class of sklearn that select for one regressor giving from parameters the best parameters according to a dictionary with the different values to try in the regressor execution.

## 3. Method to obtain the best results
The combination of the different strategies and changes in main decisions, with the scores and the different predictions, the result of lasts given us the most suitable strategy to attack the problem as:

1. Add the RandomForest regressor to give better results.
2. Interpolate method and filling the rest with means was the best strategy (2.d)
3. The points 3, 4 and 5 of the last doesn’t touch.

**The best final score is 26.0385.**
