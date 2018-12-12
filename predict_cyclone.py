#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Dec  4 10:41:31 2018

@author: Sébastien Gardoll
"""
import os.path as path

import multiprocessing as mp
from multiprocessing import Pool
import ctypes

import common
import extraction_utils as utils
from common import Era5

import csv

import numpy as np
import pandas as pd

import keras
from keras.models import load_model

import configparser

import time
start = time.time()

def normalize_dataset(chan_array, variable, netcdf_dataset, time_step, mean, stddev):
  if variable.value.level is None:
    data = netcdf_dataset[variable.value.str_id][time_step]
  else:
    level_index = variable.value.index_mapping[variable.value.level]
    data = netcdf_dataset[variable.value.str_id][time_step][level_index]
  unsharable_norm_dataset = (data - mean)/stddev
  np.copyto(dst=chan_array, src=unsharable_norm_dataset, casting='no')

def extract_region(img):
  (id, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx) = img
  for variable in Era5:
    nc_dataset = _NORMALIZED_DATASET[variable.value.num_id]
    dest_array = _CHANNELS[variable.value.num_id][id]
    np.copyto(dst=dest_array, src=nc_dataset[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx], casting='no')

def open_cyclone_db():
  cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,
    f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv')
  cyclone_db_file = open(cyclone_db_file_path, 'r')
  cyclone_dataframe = pd.read_csv(cyclone_db_file, sep=',', header=0, index_col=0, na_values='')
  cyclone_db_file.close()
  return cyclone_dataframe

config_file_path = path.join(common.SCRIPT_DIR_PATH, 'prediction.ini')
config = configparser.ConfigParser()
config.read(config_file_path)

# TODO Manage an already computed stack of images. TODO
# if (len(sys.argv) > 1) and (sys.argv[1].strip()):
#

# Settings
year      = int(config['date']['year'])
month     = int(config['date']['month'])
day       = int(config['date']['day'])
time_step = int(config['date']['time_step'])

lat_max = float(config['region']['lat_max'])
lat_min = float(config['region']['lat_min'])
lon_max = float(config['region']['lon_max'])
lon_min = float(config['region']['lon_min'])

file_prefix    = config['model']['file_prefix']
threshold_prob = float(config['model']['threshold_prob'])

nb_proc  = int(config['sys']['nb_proc'])
is_debug = bool(config['sys']['is_debug'])

# set data_format to 'channels_last'
keras.backend.set_image_data_format('channels_last')

# Checkings

if lat_max < lat_min:
  print('> [ERROR] latitude input is not coherent')
  exit(-1)

if lon_max < lon_min:
  print('> [ERROR] longitude input is not coherent')
  exit(-1)

# Open the cyclone db.
cyclone_dataframe = open_cyclone_db()

# Check if there is any cyclone for the given settings.

existing_cyclones = cyclone_dataframe[(cyclone_dataframe.year == year) &
  (cyclone_dataframe.month == month) &
  (cyclone_dataframe.day == day) &
  (cyclone_dataframe.time_step == time_step)]

if existing_cyclones.empty:
  print('> [WARN] the selected region doesn\'t have any cyclone for the given\
 time period (year: {year} ; month: {month} ; day: {day} ; time_step: {time_step})')
else:
  nb_cyclones = len(existing_cyclones)
  print(f'> found {nb_cyclones} cyclone(s) for the given time period\
 (year: {year} ; month: {month} ; day: {day} ; time_step: {time_step})')

if is_debug:
  intermediate_time_1 = time.time()
  formatted_time =common.display_duration((intermediate_time_1-start))
  print(f'  > intermediate processing time: {formatted_time}')

print(f'> opening netcdf files (year: {year}; month: {month})')
netcdf_dict = utils.build_dataset_dict(year, month)

if is_debug:
  intermediate_time_2 = time.time()
  formatted_time =common.display_duration((intermediate_time_2-intermediate_time_1))
  print(f'  > intermediate processing time: {formatted_time}')

stats_filename = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{common.STATS_FILE_POSTFIX}.csv'
stats_dataframe_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                                      stats_filename)
# Allocation of the datasets which values are normalized.
for variable in Era5:
  if variable.value.level is None:
    shape = netcdf_dict[variable][variable.value.str_id][time_step].shape
    break

# Making use of numpy array backended by ctypes shared array has been successfully
# tested while in multiprocessing context.
_NORMALIZED_DATASET = np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(ctypes.c_float,
  shape[1]), shape[0]), len(Era5)))

print(f'> loading netcdf files and normalizing them ({stats_filename})')
with open (stats_dataframe_file_path, 'r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',', lineterminator='\n')
  next(csv_reader) # Skip the header.
  # Statistics are ordered the same as the ids of ERA5 enums.
  for variable in Era5:
    (mean, stddev) = next(csv_reader)
    mean = float(mean)
    stddev = float(stddev)
    normalize_dataset(_NORMALIZED_DATASET[variable.value.num_id],
                      variable, netcdf_dict[variable],
                      time_step, mean, stddev)

if not is_debug:
  for dataset in netcdf_dict.values():
    dataset.close()

if is_debug:
  intermediate_time_3 = time.time()
  formatted_time =common.display_duration((intermediate_time_3-intermediate_time_2))
  print(f'  > intermediate processing time: {formatted_time}')

# Round lat&lon so as to compute synchronize with ERA5 resolution.
rounded_lat_max = common.round_nearest(lat_max, common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
rounded_lat_min = common.round_nearest(lat_min, common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
rounded_lon_max = common.round_nearest(lon_max, common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
rounded_lon_min = common.round_nearest(lon_min, common.LON_RESOLUTION, common.NUM_DECIMAL_LON)

if rounded_lat_max != lat_max or \
   rounded_lat_min != lat_min or \
   rounded_lon_max != lon_max or \
   rounded_lon_min != lon_min:
  print('> location input has been rounded')

# Translate lat&lon into index of array.
latitude_indexes  = np.load(path.join(common.DATASET_PARENT_DIR_PATH,
                                      'latitude_indexes.npy')).item()
longitude_indexes = np.load(path.join(common.DATASET_PARENT_DIR_PATH,
                                      'longitude_indexes.npy')).item()
# Min&max are inverted for latitude.
lat_max_idx = latitude_indexes[rounded_lat_min]
lat_min_idx = latitude_indexes[rounded_lat_max]
lon_max_idx = longitude_indexes[rounded_lon_max]
lon_min_idx = longitude_indexes[rounded_lon_min]

'''
# DEBUG
print(f'lat_min_idx: {lat_min_idx}')
print(f'lat_max_idx: {lat_max_idx}')
print(f'lon_min_idx: {lon_min_idx}')
print(f'lon_max_idx: {lon_max_idx}')
'''

# Chunks the given region into multiple subregion <=> images.
# Tuple composition: (id, (lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx),
# (lat_min, lat_max, lon_min, lon_max)).
print(f'> chunking the selected region (lat min: {rounded_lat_min} ; \
lat max: {rounded_lat_max} ; lon min: {rounded_lon_min} ; lon max: {rounded_lon_max})')
id_counter = 0
index_list = []
image_list = []
current_lat_min_idx = lat_min_idx
current_lat_max     = rounded_lat_max # Latitude indexes are inverted.
while current_lat_min_idx < lat_max_idx:
  current_lat_max_idx = current_lat_min_idx + common.Y_RESOLUTION
  current_lon_min_idx = lon_min_idx
  current_lat_min     = current_lat_max - common.LAT_FRAME
  current_lon_min     = rounded_lon_min
  while True:
    current_lon_max_idx = current_lon_min_idx + common.X_RESOLUTION
    current_lon_max     = current_lon_min + common.LON_FRAME
    index_list.append((id_counter, current_lat_min_idx, current_lat_max_idx,
      current_lon_min_idx, current_lon_max_idx))
    image_list.append([current_lat_min, current_lat_max, current_lon_min, current_lon_max])
    current_lon_min_idx = current_lon_min_idx + 1
    current_lon_min     = current_lon_min + common.LON_RESOLUTION
    id_counter = id_counter + 1
    if current_lon_min_idx > lon_max_idx:
      current_lat_min_idx = current_lat_min_idx + 1
      current_lat_max     = current_lat_max - common.LAT_RESOLUTION
      break

image_df_colums = {'lat_min': np.float32,
                   'lat_max': np.float32,
                   'lon_min': np.float32,
                   'lon_max': np.float32}
# Appending rows one by one in the while loop takes far more time then this.
image_df = pd.DataFrame(data=image_list, columns=image_df_colums.keys())
# Specify the schema.
image_df = image_df.astype(dtype = image_df_colums)
del image_list

if is_debug:
  intermediate_time_4 = time.time()
  formatted_time =common.display_duration((intermediate_time_4-intermediate_time_3))
  print(f'  > intermediate processing time: {formatted_time}')

# Allocation of the channels.
# Making use of numpy array backended by ctypes shared array has been successfully
# tested while in multiprocessing context.
_CHANNELS = np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(
  ctypes.ARRAY(ctypes.c_float, common.Y_RESOLUTION), common.X_RESOLUTION), id_counter),
  len(Era5)))

print(f'> extracting the {id_counter} subregions (proc: {nb_proc})')
with Pool(processes = nb_proc) as pool:
  pool.map(extract_region, index_list)

if is_debug:
  intermediate_time_5 = time.time()
  formatted_time =common.display_duration((intermediate_time_5-intermediate_time_4))
  print(f'  > intermediate processing time: {formatted_time}')

print('> stacking the channels')
tensor = np.stack(_CHANNELS, axis=3)

if is_debug:
  intermediate_time_6 = time.time()
  formatted_time =common.display_duration((intermediate_time_6-intermediate_time_5))
  print(f'  > intermediate processing time: {formatted_time}')

if not is_debug:
  # TODO unique name. Add name into settings file ?
  tensor_filename = f'{file_prefix}_prediction_tensor.npy'
  file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH, tensor_filename)
  print(f'> saving the tensor on disk ({tensor_filename})')
  np.save(file=file_path, arr=tensor, allow_pickle=True)

if is_debug:
  intermediate_time_7 = time.time()
  formatted_time =common.display_duration((intermediate_time_7-intermediate_time_6))
  print(f'  > intermediate processing time: {formatted_time}')

cnn_filename = f'{file_prefix}_{common.CNN_FILE_POSTFIX}.h5'
cnn_file_path = path.join(common.CNN_PARENT_DIR_PATH, cnn_filename)
print(f'> loading the CNN model ({cnn_filename})')
model = load_model(cnn_file_path)

print('> compute prediction of the subregions')
# Compute the probabilities.
y_pred_prob_npy = model.predict(tensor, verbose=1)
# Keep only the probabilities.
y_pred_prob_npy = np.delete(y_pred_prob_npy, obj=0, axis=1).squeeze()

if is_debug:
  intermediate_time_8 = time.time()
  formatted_time =common.display_duration((intermediate_time_8-intermediate_time_7))
  print(f'  > intermediate processing time: {formatted_time}')

print('> computing results')

# True corresponds to a cyclone.
class_func = np.vectorize(lambda prob: True if prob >= threshold_prob else False)
y_pred_class_npy = np.apply_along_axis(class_func, 0, y_pred_prob_npy)

y_pred_prob = pd.DataFrame(data=y_pred_prob_npy, columns=['prob'])
y_pred_class = pd.DataFrame(data=y_pred_class_npy, columns=['is_cyclone'])

# Concatenate the data frames.
image_df = pd.concat((image_df, y_pred_prob, y_pred_class), axis=1)
cyclone_images_df = image_df[image_df.is_cyclone == True]

if not cyclone_images_df.empty:
  print(f'  > model has predicted {len(cyclone_images_df)} cyclone(s)')
  filename = f'{file_prefix}_{year}_{month}_{day}_{time_step}_{common.PREDICTION_FILE_POSTFIX}.csv'
  print(f'> saving the {filename} on disk')
  no_cyclone_dataframe_file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH,
                                           filename)
  cyclone_images_df.to_csv(no_cyclone_dataframe_file_path, sep=',',
                           na_rep='', header=True, index=True,
                           index_label='id', encoding='utf8',
                           line_terminator='\n')
else:
  print('  > model has NOT predicted any cyclone')

if is_debug:
  intermediate_time_9 = time.time()
  formatted_time =common.display_duration((intermediate_time_9-intermediate_time_8))
  print(f'  > intermediate processing time: {formatted_time}')

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')

# DEBUG

# 2000,8,6,0,HU,14.5,-33.2
# => lat: 10.5 -> 18.5
#    lon: -37.25 -> -29.25
debug_lat     = 14.5
debug_lon     = -33.25
variable      = Era5.MSL

debug_lat_min = debug_lat - common.HALF_LAT_FRAME
debug_lat_max = debug_lat + common.HALF_LAT_FRAME
debug_lon_min = debug_lon - common.HALF_LON_FRAME
debug_lon_max = debug_lon + common.HALF_LON_FRAME

record = image_df[(image_df.lat_min == debug_lat_min) & (image_df.lat_max == debug_lat_max) & (image_df.lon_min == debug_lon_min) & (image_df.lon_max == debug_lon_max)]
print(record)
#       lat_min  lat_max  lon_min  lon_max          prob  is_cyclone
#30537     10.5     18.5   -37.25   -29.25  3.056721e-10       False
debug_id = record.index[0]

from matplotlib import pyplot as plt
region = _CHANNELS[variable.value.num_id][debug_id]
plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
plt.show()

import extraction_utils as utils
region = utils.extract_region(netcdf_dict[variable], variable, day, time_step, debug_lat, debug_lon)
plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
plt.show()

# The images must be the same, even if the image from the _CHANNELS is based on
# normalized values from netcdf file.