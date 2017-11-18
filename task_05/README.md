# Task 05
    
## Task Goal.
The task goal is to obtain the most relevants features of each city with all data, with the files 'dengue_training_features' and 'dengue_training_labels'.
For this task we are going to use the learned methods.

## Load Data.
Firstly, we load the data, but we are goint to do this twice because we have two cities to analyse. In addition, in this case we just are going to exclude 'week_start_date' and later we are going to remove the year.
We reuse with the same mode than the before tasks. 

## Features selection.

### Detecting and remove Outliers.
We detecting the outliers to removing it, to better analyse the data.

To detect it, we make a hierarchical clustering whit the 'single' method because it is the best outliers detector,  and distance like criterion.

* __San Juan:__

![herarchial_san_juan][1]
![initial_san_juan][2] ![final_san_juan][3]


* __Iquitos:__

![herarchial_iquitos][4]
![initial_iquitos][5] ![final_iquitos][6]

The maximum distance is obtained by analyzing the values returned by the dendrogram.
The maximum cut will be established dynamically, it will be established by means of the following expression:

__max_cut = cut - (cut * (1 - avSim))__
    
    * Where avSim is the average distance calculated in the similarity matrix.
    * Cut is the maximun distance between groups.

Knowing the above parameters, we establish percentages with respect to the total of the elements of the data set. If the studied group supposes less than 2% of the total will be eliminated, if it is less than 10% it will be reanalyzed more thoroughly, if not, said group is not an outlier.
 
### Cross Validation.
When we have eliminated the outliers and with the created groups, we will use a characteristic of each group, it will say the same thing because it is related and because we do not make more noise than necessary.

* __San Juan:__

![correlation_san_juan][7] ![max_deth_san_juan][8]

* __Iquitos:__

![correlation_iquitos][9] ![max_deth_iquitos][10]

### Decision Tree
To select the important features we will do a __"cross validation"__ with a __decision tree__ with which we are going to know some relevants features.

The resulting decision trees are these:

* __San Juan:__

 ![decision_tree_san_juan][11]


* __Iquitos:__

![decision_tree_iquitos][12]


## Conclusion.

The most relevants features to Iquitos are:
    'reanalysis_specific_humidity_g_per_kg' and 'weekofyear'
And to San Juan are: 
    'reanalysis_specific_humidity_g_per_kg' and 'reanalysis_avg_temp_k'

## Extra:

### Analyzing features (features cluster).
This is not important in this step but thanks to the elimination of outliers we can see thanks to dendrogram how similar the characteristics are, making a better study of these groups, in the future, a better smoothing of the data could be done when these they are null. (San Juan and Iquitos are similar)

![feature_cluster_san_juan][13]
![feature_cluster_iquitos][14]

[1]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/outliers/dendrogram_sanjuan.png
[2]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/outliers/initial_scatter_plot_sanjuan.png
[3]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/outliers/final_scatter_plot_sanjuan.png
[4]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/outliers/dendrogram_iquitos.png
[5]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/outliers/initial_scatter_plot_iquitos.png
[6]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/outliers/final_scatter_plot_iquitos.png
[7]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/cross_validation/correlation_sanjuan.png
[8]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/cross_validation/max_depth_sanjuan.png
[9]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/cross_validation/correlation_iquitos.png
[10]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/cross_validation/max_depth_iquitos.png
[11]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/decision_tree_mse_iq.png
[12]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/decision_tree_mse_iq.png
[13]:https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/features_cluster/features_cluster_sanjuan.png
[14]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_05/images/features_cluster/features_cluster_iquitos.png
