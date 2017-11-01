## Task 01

#### 1. Data extract:

Before starting to data recovery which have been selected, we are willing to analyze different features, considering that goals of the tasks are the existent correlation between them.

We have chosen to delete four features for the correlation analyze and PCA calculation, “city”, "year", "Weekofyear" and “week_start_date”, deb to it doesn’t provide relevant information about data correlation.

In the follow list we are going to show existent features:

|  |  |
| -- | -- |
| 1. city. | 13. reanalysis_max_air_temp_k. |
| 2. year. | 14. reanalysis_min_air_temp_k. |
| 3. Weekofyear. | 15. reanalysis_precip_amt_kg_per_m2. |
| 4. week_start_date. | 16. reanalysis_relative_humidity_percent. |
| 5. ndvi_ne. | 17. reanalysis_sat_precip_amt_mm. |
| 6. ndvi_nw. | 18. reanalysis_specific_humidity_g_per_kg. |
| 7. ndvi_se. | 19. reanalysis_tdtr_k. |
| 8. ndvi_sw. | 20. station_avg_temp_c. |
| 9. precipitation_amt_mm. | 21. station_diur_temp_rng_c. |
| 10. reanalysis_air_temp_k. | 22. station_max_temp_c. |
| 11. reanalysis_avg_temp_k. | 23. station_min_temp_c. |
| 12. reanalysis_dew_point_temp_k. | 24. station_precip_mm. |

For the data extraction we have selected the proportionated scripts which we have adapted to specification: City of ‘San Juan’, denoted to ‘sj’, years between 2004 and 2010, including both.
We have focused the extraction by two ways:

1. We accept all examples inside data target, if there are some features without data, it will be completed NaN value.
2. If the example (row) have some areas without information, it will be deleted for calculating the PCA.

## 2) Extract the correlation among features and obtain conclusions.

Considering that the name of the different features, a priori, It is easy to see existent difference between correlation types of features, the most positives and nearby to 1, for example “reanalysis_avg_temp_k” with “reanalysis_sat_precip_amt_mm”.

With the different approaches we can see few visual difference in the grade of correlation features.
![Correlation with matplotlib][1] 
![Correlation with seaborn][2]


[1]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_01/images/Correlation_matplotlib.png?raw=true
[2]: https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_01/images/Correlation_seaborn.png?raw=true

We can see how features [ndvi_nw, ndvi_se] don’t have too much correlation, meanwhile features [reanalysis_air_temp_k, reanalysis_avg_temp_k, reanalysis_dew_point_temp_k,reanalysis_max_air_temp_k] have too much correlation. Also, we can obvserve how [reanalysis_sat_precip_amt_mm, reanalysis_tdtr_k, station_diur_temp_rng_c, station_max_temp_c] have high correlation. The rest of features have some correlation but not so much.

## 3) Execute PCA and plot the results
It can be observed how everyone is so close but it can appreciate week’s (22 – 2004) and (6 – 2005) are the most unequal (outlier) and some group of them like weeks (2007-44), (47 – 2004), (2004 – 52).

![PCA](https://github.com/grego1201/MACHINE-LEARNING-TECHNIQUES/blob/master/task_01/images/PCA.png?raw=true)

## Experience
However, if we can understand the finality of task and steps which we should take it to obtain results which are a great support, if we do not have enough knowledge about correlation concepts and which is the utility about calculate the “Principal component analysis (PCA)” we can’t make a good interpretation.
Our experience has some ups and downs because at the beginning of the session we did it with a lot of enthusiasm, thinking that it will be easy, but when we thought we had finished some problems appears, like discriminate some data, which we started to work it.

