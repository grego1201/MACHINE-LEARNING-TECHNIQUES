# -*- coding: utf-8 -*-
"""

Modified on Wed Sep 27 17:00:48 2017 by:    
    GONZALO PEREZ FERNANDEZ.
    GREGORIO BALDOMERO PATINO ESTEO.
    SERGIO FERNANDEZ GARCIA.
"""

import csv

f = open("../data/dengue_features_train (copia).csv", 'rt')
city = 'sj'
range_years = range(2004,2011)

all_weeks = []
select_weeks =[]
arithmetic_means = []
heads = []

total = 0
total_in = 0
total_out = 0

smoother_data_row = []

try:
    reader = csv.reader(f)
    
    for row in reader:
        newest = row[0]
        if newest == 'city':
            # No include 'city' column (feature)
            heads.append(row[1:3]+row[4:])
        elif newest == city:
            if ( int(row[1]) in range_years ):      
                all_weeks.append(row[1:3]+row[4:])
                
                if not ('' in row):
                    # No include 'city' column (feature) and 
                    # rows whit any empty feature
                    select_weeks.append(row[1:3]+row[4:])
                    total_in += 1
                else:
                    total_out += 1
                    smoother_data_row.append(row[1:3]+row[4:])
                    #examples.append(total_in+total_out+1)
                    
finally:
    f.close()
    
for i in range(len(all_weeks[0])):
    arithmetic_means.append(0)

for i in range(len(all_weeks)):
    for y in range(len(all_weeks[i])):
        if not (all_weeks[i][y].rfind('')==0):
            arithmetic_means[y]=(arithmetic_means[y]+float(all_weeks[i][y]))/2


for i in range(len(all_weeks)):
    for y in range(len(all_weeks[i])):
        if (all_weeks[i][y].rfind('')==0):
            all_weeks[i][y]=arithmetic_means[y]

total = total_in + total_out

print("Data for: Elimination technique by \"empty features\":")
print("Loads examples: {}".format(total_in))
print("Excluded examples: {}".format(total_out))
print("Total examples: {}".format(total))


# write the results in a csv files
f1 = open("../data/dengue_pivot_select.csv",'wt')
f2 = open("../data/dengue_pivot_mean.csv",'wt')
f3 = open("../data/dengue_to_smooth.csv",'wt')

try:
    writer1 = csv.writer(f1, lineterminator='\n')
    writer1.writerows(heads)
    writer1.writerows(select_weeks)
    
    writer2 = csv.writer(f2, lineterminator='\n')
    writer2.writerows(heads)
    writer2.writerows(all_weeks)
    
        
    writer3 = csv.writer(f3, lineterminator='\n')
    writer3.writerows(heads)
    writer3.writerows(smoother_data_row)
    
finally:
    
    f1.close()
    f2.close()
    f3.close()

