# -*- coding: utf-8 -*-
"""

Modified on Wed Sep 27 17:00:48 2017 by:    
    GONZALO PEREZ FERNANDEZ.
    GREGORIO BALDOMERO PATINO ESTEO.
    SERGIO FERNANDEZ GARCIA.
"""

import csv


f = open("data/dengue_features_train.csv", 'rt')
weeks = []
week = ""
try:
    reader = csv.reader(f)
    data = None
    for row in reader:
        newest = row[0]
        if newest == 'sj':
          # process the indentation spaces
            if (int(row[1]) > 2003 and int(row[1]) < 2011): 
                if not '' in row:
                    weeks.append(row[1:])
                    data = []
                #data.append(newest)
                #week = newest
                   
finally:
    f.close()

#weeks.append(data)    
#print weeks
# write the results in a csv file

f = open("data/dengue_pivot.csv",'wt')
try:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(('1year', '2weekofyear', '3ndvi_ne',
                     '4vi_nw','5ndvi_se', '6ndvi_sw', '7precipitation_amt_mm',
                     '8reanalysis_air_temp_k', '9reanalysis_avg_temp_k',
                     '10reanalysis_dew_point_temp_k', '11reanalysis_max_air_temp_k',
                     '12reanalysis_min_air_temp_k',
                     '13reanalysis_precip_amt_kg_per_m2',
                     '14nalysis_relative_humidity_percent',
                     '15reanalysis_sat_precip_amt_mm',
                     '16reanalysis_specific_humidity_g_per_kg',
                     '17reanalysis_tdtr_k', '18station_avg_temp_c', 
                     '19station_diur_temp_rng_c', '20station_max_temp_c',
                     '21station_min_temp_c',    '22station_precip_mm'
                     
))
    writer.writerows(weeks)
finally:
    f.close()

