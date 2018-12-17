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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import keras
from keras.models import load_model

import configparser

import time
start = time.time()
previous_intermediate_time = start

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

def compute_min_max(lat, lon):
  lat_min = common.round_nearest((lat - common.HALF_LAT_FRAME), common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
  lat_max = common.round_nearest((lat + common.HALF_LAT_FRAME), common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
  lon_min = common.round_nearest((lon - common.HALF_LON_FRAME), common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
  lon_max = common.round_nearest((lon + common.HALF_LON_FRAME), common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
  return (lat_min, lat_max, lon_min, lon_max)

def format_record(idx, record):
  lat = record['lat']
  lon = record['lon']
  (lat_min, lat_max, lon_min, lon_max) = compute_min_max(lat, lon)
  return f'  > id: {idx} ; lat = {lat} ; lon = {lon} ; lat_min = {lat_min} ; lat_max = {lat_max} ; lon_min = {lon_min} ; lon_max = {lon_max}'

def display_intermediate_time():
  if is_debug:
    global previous_intermediate_time
    intermediate_time = time.time()
    formatted_time =common.display_duration((intermediate_time-previous_intermediate_time))
    previous_intermediate_time = intermediate_time
    print(f'  > intermediate processing time: {formatted_time}')

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
save_tensor = bool(config['sys']['save_tensor'])

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

recorded_cyclones = cyclone_dataframe[(cyclone_dataframe.year == year) &
  (cyclone_dataframe.month == month) &
  (cyclone_dataframe.day == day) &
  (cyclone_dataframe.time_step == time_step)]

if recorded_cyclones.empty:
  print('> [WARN] the selected region doesn\'t have any cyclone for the given\
 time period (year: {year} ; month: {month} ; day: {day} ; time_step: {time_step})')
else:
  nb_cyclones = len(recorded_cyclones)
  print(f'> found {nb_cyclones} cyclone(s) for the given time period\
 (year: {year} ; month: {month} ; day: {day} ; time_step: {time_step})')
  for idx, record in recorded_cyclones.iterrows():
    print(format_record(idx, record))

display_intermediate_time()

print(f'> opening netcdf files (year: {year}; month: {month})')
netcdf_dict = utils.build_dataset_dict(year, month)

display_intermediate_time()

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

time_index = utils._compute_time_index(day, time_step)

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
                      time_index, mean, stddev)
if not is_debug:
  for dataset in netcdf_dict.values():
    dataset.close()

display_intermediate_time()

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
    image_list.append([(current_lat_min+common.HALF_LAT_FRAME),
                       (current_lon_min+common.HALF_LON_FRAME),
                       current_lat_min, current_lat_max, current_lon_min,
                       current_lon_max])
    current_lon_min_idx = current_lon_min_idx + 1
    current_lon_min     = current_lon_min + common.LON_RESOLUTION
    id_counter = id_counter + 1
    if current_lon_min_idx > lon_max_idx:
      current_lat_min_idx = current_lat_min_idx + 1
      current_lat_max     = current_lat_max - common.LAT_RESOLUTION
      break

image_df_colums = {'lat'     : np.float32,
                   'lon'     : np.float32,
                   'lat_min' : np.float32,
                   'lat_max' : np.float32,
                   'lon_min' : np.float32,
                   'lon_max' : np.float32}
# Appending rows one by one in the while loop takes far more time then this.
image_df = pd.DataFrame(data=image_list, columns=image_df_colums.keys())
# Specify the schema.
image_df = image_df.astype(dtype = image_df_colums)
del image_list

display_intermediate_time()

# Allocation of the channels.
# Making use of numpy array backended by ctypes shared array has been successfully
# tested while in multiprocessing context.
_CHANNELS = np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(
  ctypes.ARRAY(ctypes.c_float, common.Y_RESOLUTION), common.X_RESOLUTION), id_counter),
  len(Era5)))

print(f'> extracting the {id_counter} subregions (proc: {nb_proc})')
with Pool(processes = nb_proc) as pool:
  pool.map(extract_region, index_list)

display_intermediate_time()

print('> stacking the channels')
tensor = np.stack(_CHANNELS, axis=3)

display_intermediate_time()

if save_tensor:
  # TODO unique name. Add name into settings file ?
  tensor_filename = f'{file_prefix}_prediction_tensor.npy'
  file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH, tensor_filename)
  print(f'> saving the tensor on disk ({tensor_filename})')
  np.save(file=file_path, arr=tensor, allow_pickle=True)
  display_intermediate_time()

print('> compute prediction of the subregions')

cnn_filename = f'{file_prefix}_{common.CNN_FILE_POSTFIX}.h5'
cnn_file_path = path.join(common.CNN_PARENT_DIR_PATH, cnn_filename)
print(f'  > loading the CNN model ({cnn_filename})')
model = load_model(cnn_file_path)


print('  > predicting categories')
# Compute the probabilities.
y_pred_prob_npy = model.predict(tensor, verbose=0)
# Keep only the probabilities.
y_pred_prob_npy = np.delete(y_pred_prob_npy, obj=0, axis=1).squeeze()

display_intermediate_time()

print('> computing results')

true_cat_serie = None
nb_missing_recorded_cyclones = 0
for idx, recorded_cyclone in recorded_cyclones.iterrows():
  lat = recorded_cyclone['lat']
  lon = recorded_cyclone['lon']
  current = (image_df.lat_min<lat) & (image_df.lat_max>lat) & (image_df.lon_min<lon) & (image_df.lon_max>lon)
  if not current.any():
    nb_missing_recorded_cyclones = nb_missing_recorded_cyclones + 1
  if true_cat_serie is not None:
    true_cat_serie = true_cat_serie | current
  else:
    true_cat_serie = current

true_cat_serie = true_cat_serie.map(arg=lambda value: 1.0 if value else 0.0)
true_cat_serie.name = 'true_cat'

# True corresponds to a cyclone.
cat_func = np.vectorize(lambda prob: 1.0 if prob >= threshold_prob else 0.0)
y_pred_cat_npy = np.apply_along_axis(cat_func, 0, y_pred_prob_npy)

y_pred_prob = pd.DataFrame(data=y_pred_prob_npy, columns=['pred_prob'])
y_pred_cat = pd.DataFrame(data=y_pred_cat_npy, columns=['pred_cat'])

# Concatenate the data frames.
image_df = pd.concat((image_df, true_cat_serie, y_pred_prob, y_pred_cat), axis=1)

cyclone_images_df = image_df[image_df.pred_cat == 1]

if not cyclone_images_df.empty:
  print(f'  > model has classified {len(cyclone_images_df)} image(s) as cyclone')
else:
  print('  > model has NOT classified any image as cyclone')

print('  > compute true labels of the subregions')

auc_model = roc_auc_score(y_true=image_df.true_cat, y_score=y_pred_prob_npy)
print(f'  > AUC: {common.format_pourcentage(auc_model)}%')

print(f'  > metrics report:')
print(classification_report(y_true=image_df.true_cat, y_pred=y_pred_cat_npy, target_names=('no_cyclones', 'cyclones')))

display_intermediate_time()

if not cyclone_images_df.empty:
  filename = f'{file_prefix}_{year}_{month}_{day}_{time_step}_{common.PREDICTION_FILE_POSTFIX}.csv'
  print(f'> saving the {filename} on disk')
  cyclone_images_file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH,
                                       filename)
  cyclone_images_df.to_csv(cyclone_images_file_path, sep=',',
                           na_rep='', header=True, index=True,
                           index_label='id', encoding='utf8',
                           line_terminator='\n')
display_intermediate_time()

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')


def test(debug_lat, debug_lon):
  # Example:
  # debug_lat     = 14.5
  # debug_lon     = -33.25
  # => lat: 10.5 -> 18.5
  #    lon: -37.25 -> -29.25

  debug_lat = common.round_nearest(debug_lat, common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
  debug_lon = common.round_nearest(debug_lon, common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
  variable      = Era5.MSL

  debug_lat_min = debug_lat - common.HALF_LAT_FRAME
  debug_lat_max = debug_lat + common.HALF_LAT_FRAME
  debug_lon_min = debug_lon - common.HALF_LON_FRAME
  debug_lon_max = debug_lon + common.HALF_LON_FRAME

  record = image_df[(image_df.lat_min == debug_lat_min) & (image_df.lat_max == debug_lat_max) & (image_df.lon_min == debug_lon_min) & (image_df.lon_max == debug_lon_max)]
  print(record)
  #       lat_min  lat_max  lon_min  lon_max          pred_prob  pred_cat
  #30537     10.5     18.5   -37.25   -29.25  3.056721e-10       False
  debug_id = record.index[0]

  print(f'indexes: {index_list[debug_id]}')
  print(f'recomputed indexes: lat_min_index = {latitude_indexes[debug_lat_max]} ; lat_max_index = {latitude_indexes[debug_lat_min]} ; lon_min_index = {longitude_indexes[debug_lon_min]} ; lon_max_index = {longitude_indexes[debug_lon_max]}')

  from matplotlib import pyplot as plt
  region = _CHANNELS[variable.value.num_id][debug_id]
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

  region = utils.extract_region(netcdf_dict[variable], variable, day, time_step, debug_lat, debug_lon)
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

  # The images must be the same, even if the image from the _CHANNELS is based on
  # normalized values from netcdf file.