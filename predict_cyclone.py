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
  numpy_wrapping = np.ctypeslib.as_array(chan_array)
  np.copyto(dst=numpy_wrapping, src=unsharable_norm_dataset, casting='no')

def extract_region(img):
  (id, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx) = img
  for variable in Era5:
    nc_dataset = normalized_dataset[variable.value.num_id]
    dest_array = channels[variable.value.num_id][id]
    x_index = 0
    for current_lat in range(lat_min_idx, lat_max_idx):
      y_index = 0
      for current_lon in range(lon_min_idx, lon_max_idx):
        dest_array[x_index][y_index] = nc_dataset[current_lat][current_lon]
        y_index = y_index + 1
      x_index = x_index + 1

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

nb_proc = int(config['sys']['nb_proc'])

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
  print('> [WARN] the selected region doesn\'t have any cyclone')

print(f'> opening netcdf files (year: {year}; month: {month})')
netcdf_dict = utils.build_dataset_dict(year, month)

stats_filename = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{common.STATS_FILE_POSTFIX}.csv'
print(f'> opening stats file {stats_filename}')
stats_dataframe_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                                      stats_filename)

# Allocation of the datasets which values are normalized.
for variable in Era5:
  if variable.value.level is None:
    shape = netcdf_dict[variable][variable.value.str_id][time_step].shape
    break

normalized_dataset = mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(ctypes.c_float,
  shape[1]), shape[0]), len(Era5))

print(f'> loading netcdf files and normalizing them')
with open (stats_dataframe_file_path, 'r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',', lineterminator='\n')
  next(csv_reader) # Skip the header.
  # Statistics are ordered the same as the ids of ERA5 enums.
  for variable in Era5:
    (mean, stddev) = next(csv_reader)
    mean = float(mean)
    stddev = float(stddev)
    normalize_dataset(normalized_dataset[variable.value.num_id],
                      variable, netcdf_dict[variable],
                      time_step, mean, stddev)

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
# Min/max is inverted for latitude.
lat_max_idx = latitude_indexes[rounded_lat_min]
lat_min_idx = latitude_indexes[rounded_lat_max]
lon_max_idx = longitude_indexes[rounded_lon_max]
lon_min_idx = longitude_indexes[rounded_lon_min]

# DEBUG
'''
print(f'lat_max_idx: {lat_max_idx}')
print(f'lat_min_idx: {lat_min_idx}')
print(f'lon_max_idx: {lon_max_idx}')
print(f'lon_min_idx: {lon_min_idx}')
'''

# Chunks the given region into multiple subregion <=> images.
# Tuple composition: (id, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx).
id_counter = 0
image_list = []
current_lat_min_idx = lat_min_idx

print(f'> chunking the selected region (lat min: {rounded_lat_min} ; \
lat max: {rounded_lat_max} ; lon min: {rounded_lon_min} ; lon max: {rounded_lon_max})')
while current_lat_min_idx < lat_max_idx:
  current_lat_max_idx = current_lat_min_idx + common.Y_RESOLUTION
  current_lon_min_idx = lon_min_idx
  while True:
    current_lon_max_idx = current_lon_min_idx + common.X_RESOLUTION
    image_list.append((id_counter, current_lat_min_idx, current_lat_max_idx, current_lon_min_idx, current_lon_max_idx))
    current_lon_min_idx = current_lon_max_idx
    id_counter = id_counter + 1
    if current_lon_min_idx > lon_max_idx:
      current_lat_min_idx = current_lat_max_idx
      break

# Allocation of the channels.
channels = mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(ctypes.ARRAY(ctypes.c_float,
  common.Y_RESOLUTION), common.X_RESOLUTION), id_counter), len(Era5))

print(f'> extracting the subregions (proc: {nb_proc})')
with Pool(processes = nb_proc) as pool:
  pool.map(extract_region, image_list)

print('> stacking the channels')
numpy_channels = np.ctypeslib.as_array(channels)
tensor = np.stack(numpy_channels, axis=3)

# TODO unique name. Add name into settings file ?
tensor_filename = f'{file_prefix}_prediction_tensor.npy'
file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH, tensor_filename)
print(f'> saving the tensor on disk ({tensor_filename})')
np.save(file=file_path, arr=tensor, allow_pickle=True)

cnn_filename = f'{file_prefix}_{common.CNN_FILE_POSTFIX}.h5'
cnn_file_path = path.join(common.CNN_PARENT_DIR_PATH, cnn_filename)
print(f'> loading the CNN model ({cnn_filename})')
model = load_model(cnn_file_path)
model.summary()

print('> compute prediction of the subregions')
# Compute the probabilities.
y_pred_prob  = model.predict(tensor, verbose=1)

# TODO
# y_pred_class = np.argmax(y_pred_prob, axis=1) # Compute the closest class.

# TODO compute the lat/lon of the subregions.

