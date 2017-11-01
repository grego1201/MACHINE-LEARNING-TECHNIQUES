# Task 02

## Main decision.
At time to realize the matrix of the distances from the similitude matrix, we opted to take the **_eucledian distance_**, because we use real values. _“Manhattan distance”_ option fall short of sense when values don’t belong to discrete variables. On the other hand, results obtain with _“Chebyshev distance”_, _a priori_, we could see they don’t be as correct as distance.

## Decision in the clustering.

To realize the group with the distance matrix in the dendrogram, we have used **_“complete method”_**, which if we compare with the rest, it is better for distribute the group, if we don’t see the _“ward method"_, which have too much groups.
Cluster hierarchic groups to realize a dispersion graphic representation was realized in **cut in 10** and the **_criterion_** used have been **_“distance”_**.

![Dendrogram][1]  ![Plot][2]

[1]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_02/images/dendrogram.png?raw=true
[2]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_02/images/plot.png?raw=true

## Explication of the results.
We have identified 5 different groups as result of operation we realize in the task, data and graphics representation, which we name with these different meanings.
  * Group 1: Low temperatures.
  * Group 2: Low precipitation.
  * Group 3: High precipitation.
  * Group 4: High temperatures.
  * Group 5: Outliers.

## [Optional] Execute the hierarchical clustering algorithm using feature as elements. Extract some conclusions.
The same sequence of steps is carried out after transposition of the matrix. The following dendrogram and scatter plot are obtained.

![Features dendrogram][3]  ![Feature Plot][4]

[3]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_02/images/dendrogram_features.png?raw=true
[4]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_02/images/plot_features.png?raw=true

Some comclusions:
 * Similar characteristics are grouped in the same cluster. (ndvi, values related to the magnitude in reanalysis and stations)
 * In reanalysis they are mainly grouped by the measure in kelvin and atmospheric pressures.
 * In station are grouped by the values of temperatures.
 * Within each of the different cluster appear some characteristic that apparently unrelated similarity with the other features that make it up.
