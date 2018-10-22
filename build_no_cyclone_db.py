#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:40:14 2018

@author: Sébastien Gardoll
"""

import numpy as np
import pandas as pd
import os.path as path

import common

import time
start = time.time()

FILE_PREFIX = '2k'
CYCLONE_DATAFRAME = None

cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                                 f'{FILE_PREFIX}_extraction_dataset.csv')
cyclone_db_file = open(cyclone_db_file_path, 'r')
CYCLONE_DATAFRAME = pd.read_csv(cyclone_db_file, sep=',', header=0, index_col=0,\
                                na_values='')
cyclone_db_file.close()
del cyclone_db_file
del cyclone_db_file_path

NO_CYCLONE_DF_COLUMNS = {'year': np.int32,\
                         'month': np.int32,\
                         'day': np.int32,\
                         'time_step': np.int32,\
                         'lat': np.float32,\
                         'lon': np.float32}

def _has_cyclone(date, time_step, lat, lon):
  records = CYCLONE_DATAFRAME.query(f'year=={date.year} and\
                                      month=={date.month}\
                                      and day=={date.day} and\
                                      time_step=={time_step}')
  if not records.empty:
    for (index, record) in records.iterrows():
      lat2 = record["lat"]
      lon2 = record["lon"]
      if common.is_overlapping(lat, lon, lat2, lon2):
        return True
    return False
  else:
    return False

def compute_no_cyclone(time, delta):
  (year, month, day, time_step, lat, lon) = time
  past = common.subtract_delta(year, month, day, delta)
  has_cyclone = _has_cyclone(past, time_step, lat, lon)
  while has_cyclone:
    time_step = time_step - 1
    if time_step < 0:
      time_step = common.TIME_SAMPLING - 1
      past = common._subtract_one_day(past)
    has_cyclone = _has_cyclone(past, time_step, lat, lon)
  return (past.year, past.month, past.day, time_step, lat, lon)

                            ######## MAIN ########

no_cyclone_dataframe = pd.DataFrame(columns=NO_CYCLONE_DF_COLUMNS.keys())

print("> computing the no cyclone records")
current_year = -1
for (index, row) in CYCLONE_DATAFRAME.iterrows():
  cyclone_year      = row["year"]
  cyclone_month     = row["month"]
  cyclone_day       = row["day"]
  cyclone_time_step = row["time_step"]
  cyclone_lat = row["lat"]
  cyclone_lon = row["lon"]
  if current_year != cyclone_year:
    current_year = cyclone_year
    print(f'  > compute year:{current_year}')
  cyclone_values = (cyclone_year, cyclone_month, cyclone_day,\
                    cyclone_time_step, cyclone_lat, cyclone_lon)
  values = compute_no_cyclone(cyclone_values, common.ONE_DAY)
  row_to_add = pd.Series(values, index=NO_CYCLONE_DF_COLUMNS.keys())
  no_cyclone_dataframe = no_cyclone_dataframe.append(row_to_add, ignore_index=True)
  values = compute_no_cyclone(cyclone_values, common.ONE_WEEK)
  row_to_add = pd.Series(values, index=NO_CYCLONE_DF_COLUMNS.keys())
  no_cyclone_dataframe = no_cyclone_dataframe.append(row_to_add, ignore_index=True)

# Specify the schema.
no_cyclone_dataframe = no_cyclone_dataframe.astype(dtype = NO_CYCLONE_DF_COLUMNS)

# Remove duplicated rows.
print(f'> number of records before removing the duplicates: {len(no_cyclone_dataframe)}')
no_cyclone_dataframe = no_cyclone_dataframe.drop_duplicates()
print(f'> number of records AFTER removing the duplicates: {len(no_cyclone_dataframe)}')

# Sort by date (month) (optimize building tensors)
print("> sorting the rows")
no_cyclone_dataframe.sort_values(by=["year", "month"], ascending = True,\
                                 inplace=True)
print("> saving the no cyclone db on disk")
no_cyclone_dataframe_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                                           f'{FILE_PREFIX}_no_cyclone_dataset.csv')
no_cyclone_dataframe.to_csv(no_cyclone_dataframe_file_path, sep = ',',\
                            na_rep = '', header = True, index = True,\
                            index_label='id', encoding = 'utf8',\
                            line_terminator = '\n')

stop = time.time()
print("spend %f seconds processing"%((stop-start)))