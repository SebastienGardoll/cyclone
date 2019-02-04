#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:08:31 2019

@author: sgardoll
"""

import os.path as path

import pandas as pd

import common

from predict_cyclone_utils import fetch_setting, open_cyclone_db, compute_recorded_cyclones
from predict_cyclone_utils import normalize_netcdf, compute_chunks, allocate_channel_array
from predict_cyclone_utils import prediction_analysis, extract_region, display_intermediate_time
from predict_cyclone_utils import open_netcdf_files, load_cnn_model, METRICS_COLUMNS, check_interval

from extraction_utils import close_dataset_dict

import time
start = time.time()

def fetch_location_vars(cyclone_location):
  year      = cyclone_location[2]
  month     = cyclone_location[3]
  day       = cyclone_location[4]
  time_step = cyclone_location[5]
  lat       = cyclone_location[7]
  lon       = cyclone_location[8]
  return (year, month, day, time_step, lat, lon)

YEAR, MONTH, DAY, TIME_STEP, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,\
FILE_PREFIX, CYCLONE_LAT_SIZE, CYCLONE_LON_SIZE, NB_PROC, IS_DEBUG,\
HAS_SAVE_RESULTS = fetch_setting()

# Process the cyclone locations of the entire month.
del DAY, TIME_STEP

# These variables are shared between the processes, thanks to start method set
# to 'fork'
cyclone_dataframe = open_cyclone_db()
CHUNK_LIST_DF, INDEX_LIST, ID_COUNTER = compute_chunks(LAT_MIN, LAT_MAX,
                                                      LON_MIN, LON_MAX)
NETCDF_DICT, SHAPE = open_netcdf_files(YEAR, MONTH)

SELECTED_CYCLONE_LOCATIONS_DF = cyclone_dataframe[(cyclone_dataframe.year == YEAR) &\
                                                  (cyclone_dataframe.month == MONTH)]

if not check_interval(SELECTED_CYCLONE_LOCATIONS_DF, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):
  exit(common.ERROR_CODE)

selected_cyclone_locations = SELECTED_CYCLONE_LOCATIONS_DF.values.tolist()

model = load_cnn_model(FILE_PREFIX)

del cyclone_dataframe

list_metrics = list()

for cyclone_location in selected_cyclone_locations:
  year, month, day, time_step, lat, lon = fetch_location_vars(cyclone_location)
  recorded_cyclones, nb_cyclones = compute_recorded_cyclones(SELECTED_CYCLONE_LOCATIONS_DF,
                                                    year, month, day, time_step)
  normalized_dataset = normalize_netcdf(FILE_PREFIX, NETCDF_DICT, SHAPE, day,
                                        time_step)
  channels_array = allocate_channel_array(ID_COUNTER)
  for img_spec in INDEX_LIST:
    extract_region(img_spec, normalized_dataset, channels_array)
  cyclone_images_df, metrics = prediction_analysis(FILE_PREFIX, channels_array,
                      recorded_cyclones, CHUNK_LIST_DF, CYCLONE_LAT_SIZE,
                      CYCLONE_LON_SIZE, nb_cyclones, model)
  list_metrics.append(metrics)

file_name = f'{FILE_PREFIX}_{YEAR}_{MONTH}_{CYCLONE_LAT_SIZE}-{CYCLONE_LON_SIZE}_{common.PREDICTION_FILE_POSTFIX}.csv'
print(f'> saving the metrics {file_name}')
file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH, file_name)

metrics_df = pd.DataFrame(data=list_metrics, columns=METRICS_COLUMNS)

metrics_df.to_csv(file_path, sep=',', na_rep='', header=True, index=True,
                  index_label='id', encoding='utf8', line_terminator='\n')

display_intermediate_time

close_dataset_dict(NETCDF_DICT)

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')
