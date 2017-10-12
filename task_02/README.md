# Task 02

## Main decision.
At time to realize the matrix of the distances from the similitude matrix, we opted to take the **_eucledian distance_**, because we use real values. _“Manhattan distance”_ option fall short of sense when values don’t belong to discrete variables. On the other hand, results obtain with _“Chebyshev distance”_, _a priori_, we could see they don’t be as correct as distance.

## Decision in the clustering.

To realize the group with the distance matrix in the dendograma, we have used **_“complete method”_**, which if we compare with the rest, it is better for distribute the group, if we don’t see the _“ward method"_, which have too much groups.
Cluster hierarchic groups to realize a dispersion graphic representation was realized in **cut in 10** and the **_criterion_** used have been **_“distance”_**.

![Dendogram][1]  ![Plot 2.1][2]

[1]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_02/images/dendogram.png?raw=true
[2]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_02/images/plot.png?raw=true

## Explication of the results.
We have identified seven different groups as result of operation we realize in the task, data and graphics representation, which we name with these different meanings.
  * Group 1: Half-high temperatures.
  * Group 2: Inconclusive. (Maybe bad measures)
  * Group 3: Low precipitation.
  * Group 4: We should look for motive because elements belong with the last weeks of April and May.
  * Group 5: Outliers
  * Group 6: Average temperatures are high.
  * Group 7: High season temperature.

#Important
During task realization we could see values in the area ndvi which don’t be between [-1.0, +1.0] and temperatures with mistakes in the data introduction, for example 25075 ºC in place of 25.075 ºC.
These mistakes are going to be detected and solved later, beginning the process to revise conclusions we saw before and changing whatever will be necessary. 
