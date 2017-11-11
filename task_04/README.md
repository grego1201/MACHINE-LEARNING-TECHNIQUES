# Task 04

## LOAD DATA.
It has been done in the same way as the previous tasks, with the same data set and eliminating the outliers that have been found (732 and 769).

This time, the values of _'total_cases'_ feature has been included in the target data, features belonging to the file _'dengue_labels_train.csv'_.
The values of _'total_cases'_ were matched considering _'city'_, _'year'_ and _'weekofyear'_ features.

## GRAPHICAL REPRESENTATIONS.

To facilitate the interpretation of the data present on the target data, after filtering them, show the histograms, density and scatter_matrix plots with the differences between features.

![Histogram][1]  

![Density][2]

These representations facilitate, in a way, the understanding of the features ratio.

![Scater Matrix][3]

## CORRELATION.

After normalization of the data using MinMax, the correlation of each of the features was calculated with the _'total_cases'_ feature, obtaining the following plot with the corresponding associated correlation values.

![Correlation][4]  

|  |  |
| -- | -- |
| __Feature__ | __R value__ |
| ndvi_ne | 0.227356 |
| __ndvi_nw__ | __0.760126__ |
| ndvi_se | 0.359821 |
| ndvi_sw | 0.390777 |
| precipitation_amt_mm | 0.658474 |
| reanalysis_air_temp_k | 0.533453 |
| reanalysis_avg_temp_k | 0.617753 |
| reanalysis_dew_point_temp_k | 0.562219 |
| reanalysis_max_air_temp_k | 0.648292 |
| reanalysis_min_air_temp_k | 0.519861 |
| __reanalysis_precip_amt_kg_per_m2__ | __0.734534__ |
| reanalysis_relative_humidity_percent | 0.60195 |
| reanalysis_sat_precip_amt_mm | -0.0701244 |
| __reanalysis_specific_humidity_g_per_kg__ | __0.782729__ |
| reanalysis_tdtr_k | 0.64281 |
| __station_avg_temp_c__ | __0.763298__ |
| station_diur_temp_rng_c | 0.676888 |
| station_max_temp_c | 0.505806 |
| station_min_temp_c | 0.386755 |
| station_precip_mm | 0.692587 |
</center>

The features, was marked in __bold__ on the table, have been selected, greater than 0.71 (in absolute value), which usually means a high or strong degree of correlation. 


## CROSS VALIDATION.
Different regression trees have been generated according to the maximum depth of these and a cross-validation has been carried out to choose the best option.

### Depth selection.
We selected the lowest value of Cross Validation, increasing the depth of the decision tree could be overfitting, we must keep in mind that the correlations of the features are above 0.71, otherwise we should take higher values or perform tests with several values to see which one fits best.

![CV Score][5]

Best MAX_DEPTH: 2


## BUILD THE MODEL.

When creating the regression model, the MSE (Mean Squared Error) criterion is by default, the maximum depth is declared at 2, the other parameters are by default.

|  |  |
| -- | -- |
| Level depth | Mean | Standard Deviation |
|  2 | 19.0044 |  +/- 14.3986135116 |
|  3 | 19.4933 |  +/- 14.847248468 |
|  4 | 20.5168 |  +/- 16.0544748648 |
|  5 | 21.5403 |  +/- 16.4651097573 |
|  6 | 21.1151 | +/- 18.455424585 |
|  7 | 22.0872 | +/- 16.9204951223 |
|  8 | 22.124  | +/- 16.2964483219 |
|  9 | 22.5326 | +/- 16.1487113534 |
| 10 | 22.4979 | +/- 15.6147082482 |
| 11 | 22.9695 | +/- 17.0185578321 |
| 12 | 23.1533 | +/- 17.4850772745 |
| 13 | 21.3258 | +/- 16.1742677446 |
| 14 | 21.688  | +/- 17.1866991109 |
| 15 | 22.7444 | +/- 17.5363523251 |
| 16 | 22.0111 | +/- 15.8740074096 |
| 17 | 22.1121 | +/- 16.4805132101 |
| 18 | 21.4454 | +/- 16.4707974135 |
| 19 | 22.483  | +/- 18.0167704447 |
| 20 | 20.9484 | +/- 16.695271736 |
| 21 | 22.2748 | +/- 16.0361343495 |
| 22 | 22.2605 | +/- 17.361780613 |
| 23 | 21.9382 | +/- 16.2117663034 |
| 24 | 22.9069 | +/- 17.9040090419 |
| 25 | 22.0069 | +/- 16.1651787651 |
| 26 | 22.9373 | +/- 17.7526872516 |
| 27 | 20.7281 | +/- 16.317407424 |
| 28 | 22.502  | +/- 17.0106518282 |
| 29 | 22.165  | +/- 16.1200614316 |

Best MAX_DEPTH: 2

To find out which features are referenced we observe relevance level of the same:

                [ RELEVANCES FEATURES ]
|  |  |
| -- | -- |
| Feature selected | Relevance |
| ndvi_nw | 0 |
| __reanalysis_precip_amt_kg_per_m2__ | __0.151773__ |
| __reanalysis_specific_humidity_g_per_kg__ | __0.848227__ |
| station_avg_temp_c | 0 |

Features with more relevance: 
        ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_precip_amt_kg_per_m2']

### Decision Tree.

![Decision Tree][6]  

[1]:
https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_04/images/hist.png
[2]:
 https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_04/images/density.png
[3]:
https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_04/images/scater_matrix.png
[4]:
 https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_04/images/correlation.png
[5]:
https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_04/images/max_depth.png?raw=true
[6]:
https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_04/images/decision_tree_mse.png
