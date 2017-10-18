# Task 03

# Load data

At the begining we load data, but removing outliers to avoid to take bad decisions and clean data. Our outliers are 21 and 38.

# Normalization and the K choice

After load of data we normalize with 'MinMaxScaler' algorithm.

We use the PCA to display data looking for the best value of the K. It has been calculated with the silohouettes coefficient, taking normalize data.

We have tested method random and method k++, but we can observe few changes. Due to it we chose k++ methopd because this algorithm observe better data than random method, and takes better decisions.
Observing silohouette graphic we chose K=2, it is due to silhouette coefficiente takes the maximum value.
![Distortion][1]  ![Silohouette][2]


# Execute clustering (k-means)

After all, with the k=2, we plotted the points visualizing them in two dimensions. We can observe that two cluster has been created. When we calculate features average, we can see that total precipitation is bigger in the group 1 than the second group.
![KMeans Groups][3]

[1]:
 https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_03/images/Distortion.png
[2]:
 https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_03/images/Silohouette.png
[3]:
https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_03/images/kgroups.png
