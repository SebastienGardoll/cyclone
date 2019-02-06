#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:46:57 2019

@author: sgardoll
"""

import os.path as path

import multiprocessing as mp

import ctypes

import common
import extraction_utils as ex_utils
from common import Era5

import csv

import numpy as np
import pandas as pd

import keras

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from keras.models import load_model

import time

                         ##### GLOBAL VARIABLES #####


PREVIOUS_INTERMEDIATE_TIME = time.time()
IS_DEBUG = False

METRICS_COLUMNS = ['auc', 'precision_no_cyclone', 'recall_no_cyclone',
                   'precision_cyclone', 'recall_cyclone', 'has_found_all_recorded_cyclone',
                   'false_positives_expected', 'false_positives_not_containing_cyclone',
                   'nb_recorded_cyclones', 'nb_missing_recorded_cyclones', 'nb_false_positives',
                   'nb_false_negatives', 'nb_true_positives', 'nb_true_negatives']

                            ##### FUNCTIONS #####


def _normalize_dataset(chan_array, variable, netcdf_dataset, time_index, mean, stddev):
  if variable.value.level is None:
    data = netcdf_dataset[variable.value.str_id][time_index]
  else:
    level_index = variable.value.index_mapping[variable.value.level]
    data = netcdf_dataset[variable.value.str_id][time_index][level_index]
  unsharable_norm_dataset = (data - mean)/stddev
  np.copyto(dst=chan_array, src=unsharable_norm_dataset, casting='no')

def extract_region(img_spec, normalized_dataset, channels_array):
  (id, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx) = img_spec
  for variable in Era5:
    nc_dataset = normalized_dataset[variable.value.num_id]
    dest_array = channels_array[variable.value.num_id][id]
    np.copyto(dst=dest_array, src=nc_dataset[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx], casting='no')

def open_cyclone_db():
  cyclone_db_file = open(common.CYCLONE_ALL_DB_FILE_PATH, 'r')
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
  if IS_DEBUG:
    global PREVIOUS_INTERMEDIATE_TIME
    intermediate_time = time.time()
    formatted_time =common.display_duration((intermediate_time-PREVIOUS_INTERMEDIATE_TIME))
    PREVIOUS_INTERMEDIATE_TIME = intermediate_time
    print(f'  > intermediate processing time: {formatted_time}')

def compute_recorded_cyclones(cyclone_dataframe, year, month, day, time_step):
  # Check if there is any cyclone for the given settings.
  recorded_cyclones = cyclone_dataframe[(cyclone_dataframe.year == year) &
    (cyclone_dataframe.month == month) &
    (cyclone_dataframe.day == day) &
    (cyclone_dataframe.time_step == time_step)]

  if recorded_cyclones.empty:
    print(f'> [WARN] the selected region doesn\'t have any cyclone for the given\
   time period (year: {year} ; month: {month} ; day: {day} ; time_step: {time_step})')
  else:
    nb_cyclones = len(recorded_cyclones)
    print(f'> found {nb_cyclones} cyclone(s) for the given time period\
   (year: {year} ; month: {month} ; day: {day} ; time_step: {time_step})')
    for idx, record in recorded_cyclones.iterrows():
      print(format_record(idx, record))

  display_intermediate_time()
  return (recorded_cyclones, nb_cyclones)

def compute_interval(dataframe):
  lat_min = dataframe['lat'].min() - common.LAT_FRAME
  lat_max = dataframe['lat'].max() + common.LAT_FRAME
  lon_min = dataframe['lon'].min() - common.LON_FRAME
  lon_max = dataframe['lon'].max() + common.LON_FRAME
  return (lat_min, lat_max, lon_min, lon_max)

def check_interval(dataframe, lat_min, lat_max, lon_min, lon_max):
    df_lat_min, df_lat_max, df_lon_min, df_lon_max = compute_interval(dataframe)
    if df_lat_min < lat_min:
      print(f'lat_min must be lesser than {df_lat_min}')
      return False
    if df_lon_min < lon_min:
      print(f'lon_min must be lesser than {df_lon_min}')
      return False
    if df_lat_max > lat_max:
      print(f'lat_max must be greater than {df_lat_max}')
      return False
    if df_lon_max > lon_max:
      print(f'lon_max must be greater than {df_lon_max}')
      return False
    return True

def open_netcdf_files(year, month):
  print(f'> opening netcdf files (year: {year}; month: {month})')
  netcdf_dict = ex_utils.build_dataset_dict(year, month)
  for variable in Era5:
    if variable.value.level is None:
      shape = netcdf_dict[variable][variable.value.str_id][0].shape
      break
  display_intermediate_time()
  return (netcdf_dict, shape)

def normalize_netcdf(file_prefix, netcdf_dict, shape, day, time_step):
  # Allocation of the datasets which values are normalized.
  # Making use of numpy array backended by ctypes shared array has been successfully
  # tested while in multiprocessing context.
  normalized_dataset = np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(ctypes.c_float,
    shape[1]), shape[0]), len(Era5)))
  time_index = ex_utils._compute_time_index(day, time_step)

  stats_filename = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{common.STATS_FILE_POSTFIX}.csv'
  print(f'> normalizing netcdf files ({stats_filename})')
  stats_dataframe_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                                        stats_filename)
  with open (stats_dataframe_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', lineterminator='\n')
    next(csv_reader) # Skip the header.
    # Statistics are ordered the same as the ids of ERA5 enums.
    for variable in Era5:
      (mean, stddev) = next(csv_reader)
      mean = float(mean)
      stddev = float(stddev)
      _normalize_dataset(normalized_dataset[variable.value.num_id],
                         variable, netcdf_dict[variable],
                         time_index, mean, stddev)
  display_intermediate_time()
  return normalized_dataset

def compute_chunks(lat_min, lat_max, lon_min, lon_max):
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
  latitude_indexes  = common.read_dict_from_csv(
                        path.join(common.DATASET_PARENT_DIR_PATH,
                                 'latitude_indexes.csv'), float, int)
  longitude_indexes = common.read_dict_from_csv(
                        path.join(common.DATASET_PARENT_DIR_PATH,
                                 'longitude_indexes.csv'), float, int)

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
  print(f'> chunking the selected region (lat min: {rounded_lat_min} ;\
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

  chunk_df_colums = {'lat'     : np.float32,
                     'lon'     : np.float32,
                     'lat_min' : np.float32,
                     'lat_max' : np.float32,
                     'lon_min' : np.float32,
                     'lon_max' : np.float32}
  # Appending rows one by one in the while loop takes far more time then this.
  chunk_list_df = pd.DataFrame(data=image_list, columns=chunk_df_colums.keys())
  # Specify the schema.
  chunk_list_df = chunk_list_df.astype(dtype = chunk_df_colums)
  display_intermediate_time()
  return (chunk_list_df, index_list, id_counter)

def fetch_setting():
  import configparser
  config_file_path = path.join(common.SCRIPT_DIR_PATH, 'prediction.ini')
  config = configparser.ConfigParser()
  config.read(config_file_path)
  # Settings
  year      = int(config['date']['year'])
  month     = int(config['date']['month'])
  day       = int(config['date']['day'])
  time_step = int(config['date']['time_step'])

  lat_max = float(config['region']['lat_max'])
  lat_min = float(config['region']['lat_min'])
  lon_max = float(config['region']['lon_max'])
  lon_min = float(config['region']['lon_min'])

  file_prefix      = config['model']['file_prefix']
  cyclone_lat_size = float(config['model']['cyclone_lat_size'])
  cyclone_lon_size = float(config['model']['cyclone_lon_size'])

  nb_proc          = int(config['sys']['nb_proc'])
  global IS_DEBUG
  IS_DEBUG         = bool(config['sys']['is_debug'])
  has_save_results = bool(config['sys']['save_results'])

  # set data_format to 'channels_last'
  keras.backend.set_image_data_format('channels_last')

  # Checkings
  if lat_max < lat_min:
    print('> [ERROR] latitude input is not coherent')
    exit(-1)

  if lon_max < lon_min:
    print('> [ERROR] longitude input is not coherent')
    exit(-1)

  return (year, month, day, time_step, lat_min, lat_max, lon_min, lon_max,
          file_prefix, cyclone_lat_size, cyclone_lon_size, nb_proc, IS_DEBUG,
          has_save_results)

# Allocation of the channels.
# Making use of numpy array backended by ctypes shared array has been successfully
# tested while in multiprocessing context.
def allocate_channel_array(id_counter):
  return np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(
  ctypes.ARRAY(ctypes.c_float, common.Y_RESOLUTION), common.X_RESOLUTION), id_counter),
  len(Era5)))

def load_cnn_model(file_prefix):
  cnn_filename = f'{file_prefix}_{common.CNN_FILE_POSTFIX}.h5'
  cnn_file_path = path.join(common.CNN_PARENT_DIR_PATH, cnn_filename)
  print(f'> loading the CNN model ({cnn_filename})')
  model = load_model(cnn_file_path)
  display_intermediate_time()
  return model


def compute_containing_region(chunk_list_df,recorded_cyclones,
                              cyclone_lat_size, cyclone_lon_size,
                              is_intersection=False):
  selected_serie = None
  nb_missing_recorded_cyclones = 0
  for idx, recorded_cyclone in recorded_cyclones.iterrows():
    lat = common.round_nearest(recorded_cyclone['lat'], common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
    lon = common.round_nearest(recorded_cyclone['lon'], common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
    lat_min = lat-cyclone_lat_size
    lat_max = lat+cyclone_lat_size
    lon_min = lon-cyclone_lat_size
    lon_max = lon+cyclone_lat_size
    if is_intersection: # Using pandas or numpy intersection could be better.
      current = (((chunk_list_df.lat_min < (lat_min)) &     \
                (chunk_list_df.lat_max < (lat_max)))    |   \
                ((chunk_list_df.lat_min > (lat_min))  &     \
                (chunk_list_df.lat_max > (lat_max))))     & \
                (((chunk_list_df.lon_min < (lon_min)) &     \
                (chunk_list_df.lon_max < (lon_max)))    |   \
                ((chunk_list_df.lon_min > (lon_min))  &     \
                (chunk_list_df.lon_max > (lon_max))))
    else:
      current = (chunk_list_df.lat_min <= (lat_min)) & \
                (chunk_list_df.lat_max >= (lat_max)) & \
                (chunk_list_df.lon_min <= (lon_min)) & \
                (chunk_list_df.lon_max >= (lon_max))
    if not current.any():
      nb_missing_recorded_cyclones = nb_missing_recorded_cyclones + 1
    if selected_serie is not None:
      # dataframe OR operator on the category of the image.
      selected_serie = selected_serie | current
    else:
      selected_serie = current
    return selected_serie, nb_missing_recorded_cyclones

def prediction_analysis(file_prefix, channels_array, recorded_cyclones,
                        chunk_list_df, cyclone_lat_size, cyclone_lon_size,
                        nb_cyclones, model):
  print('> stacking the channels')
  tensor = np.stack(channels_array, axis=3)

  display_intermediate_time()

  print('> compute prediction of the subregions')

  print('  > predicting categories')
  # Compute the probabilities.
  y_pred_prob_npy = model.predict(tensor, verbose=0)
  # Keep only the probabilities of the category cyclone (see the roc_auc_score help).
  y_pred_cyclone_prob_npy = np.delete(y_pred_prob_npy, obj=0, axis=1).squeeze()

  display_intermediate_time()

  print('> computing results')

  # Compute the true label of the subregions based on Hurdat2.
  # If a subregion containes a cyclone location (from Hurdat2),
  # then this subregion gets an 1.0 (cyclone), otherwise 0.0 (no cyclone).
  print(f'  > compute true labels of the subregions (cyclone size = {cyclone_lat_size} x {cyclone_lon_size})')
  true_cat_serie, nb_missing_recorded_cyclones = \
    compute_containing_region(chunk_list_df, recorded_cyclones,
                              cyclone_lat_size, cyclone_lon_size)

  true_cat_serie = true_cat_serie.map(arg=lambda value: common.CYCLONE_LABEL if value else common.NO_CYCLONE_LABEL)
  true_cat_serie.name = 'true_cat'

  # Convert the probabilities into the class based on the higher probability.
  # Class 0 for no cyclone, 1 for cyclone.
  y_pred_cat_npy = np.argmax(y_pred_prob_npy, axis=1)

  y_pred_cyclone_prob_df = pd.DataFrame(data=y_pred_cyclone_prob_npy, columns=['pred_prob'])
  y_pred_cat_df = pd.DataFrame(data=y_pred_cat_npy, columns=['pred_cat'])

  # Concatenate the data frames.
  chunk_list_df = pd.concat((chunk_list_df, true_cat_serie, y_pred_cyclone_prob_df, y_pred_cat_df), axis=1)

  cyclone_images_df = chunk_list_df[chunk_list_df.pred_cat == 1]

  if not cyclone_images_df.empty:
    print(f'  > model has classified {len(cyclone_images_df)}/{len(chunk_list_df[chunk_list_df.true_cat == 1])} images as cyclone')
  else:
    print('  > model has NOT classified any image as cyclone')

  len_classified_cyclones = nb_cyclones-nb_missing_recorded_cyclones
  print(f'  > model found {len_classified_cyclones}/{nb_cyclones} recorded cyclone(s)')

  false_positives = chunk_list_df[(chunk_list_df.pred_cat == common.CYCLONE_LABEL) & (chunk_list_df.true_cat == common.NO_CYCLONE_LABEL)]
  len_false_positives = len(false_positives)
  print(f'  > model has {len_false_positives} false positives')

  false_negatives = chunk_list_df[(chunk_list_df.pred_cat == common.NO_CYCLONE_LABEL) & (chunk_list_df.true_cat == common.CYCLONE_LABEL)]
  len_false_negatives = len(false_negatives)
  print(f'  > model has {len_false_negatives} false negatives')

  true_positives = chunk_list_df[(chunk_list_df.pred_cat == common.CYCLONE_LABEL) & (chunk_list_df.true_cat == common.CYCLONE_LABEL)]
  len_true_positives = len(true_positives)
  print(f'  > model has {len_true_positives} true positives')

  true_negatives = chunk_list_df[(chunk_list_df.pred_cat == common.NO_CYCLONE_LABEL) & (chunk_list_df.true_cat == common.NO_CYCLONE_LABEL)]
  len_true_negatives = len(true_negatives)
  print(f'  > model has {len_true_negatives} true negatives')

  false_positives_near_cyclone, tmp = \
    compute_containing_region(false_positives, recorded_cyclones,
                              cyclone_lat_size, cyclone_lon_size, True)

  len_false_positives_near_cyclone = len(false_positives_near_cyclone)

  len_false_positives_not_near_cyclone = len_false_positives - len_false_positives_near_cyclone

  print(f'  > number of false positives NOT intersecting a cyclone zone: {len_false_positives_not_near_cyclone}')

  auc_model = roc_auc_score(y_true=chunk_list_df.true_cat, y_score=y_pred_cyclone_prob_npy)
  print(f'  > AUC: {common.format_pourcentage(auc_model)}%')

  print(f'  > metrics report:\n')
  print(classification_report(y_true=chunk_list_df.true_cat, y_pred=y_pred_cat_npy, target_names=('no_cyclones', 'cyclones')))

  precision_cyclone    = precision_score(y_true=chunk_list_df.true_cat, y_pred=y_pred_cat_npy, pos_label=common.CYCLONE_LABEL)
  precision_no_cyclone = precision_score(y_true=chunk_list_df.true_cat, y_pred=y_pred_cat_npy, pos_label=common.NO_CYCLONE_LABEL)
  recall_cyclone    = recall_score(y_true=chunk_list_df.true_cat, y_pred=y_pred_cat_npy, pos_label=common.CYCLONE_LABEL)
  recall_no_cyclone = recall_score(y_true=chunk_list_df.true_cat, y_pred=y_pred_cat_npy, pos_label=common.NO_CYCLONE_LABEL)

  metrics = (auc_model, precision_no_cyclone, recall_no_cyclone, precision_cyclone,
             recall_cyclone, nb_missing_recorded_cyclones == 0,
             len_false_positives_not_near_cyclone == 0,
             len_false_positives_not_near_cyclone,
             nb_cyclones, nb_missing_recorded_cyclones, len_false_positives,
             len_false_negatives, len_true_positives, len_true_negatives)

  display_intermediate_time()

  return cyclone_images_df, metrics


def save_results(cyclone_images_df, file_prefix, year, month, day, time_step):
  if not cyclone_images_df.empty and save_results:
    filename = f'{file_prefix}_{year}_{month}_{day}_{time_step}_{common.PREDICTION_FILE_POSTFIX}.csv'
    print(f'> saving the {filename} on disk')
    cyclone_images_file_path = path.join(common.PREDICT_TENSOR_PARENT_DIR_PATH,
                                         filename)
    cyclone_images_df.to_csv(cyclone_images_file_path, sep=',',
                             na_rep='', header=True, index=True,
                             index_label='id', encoding='utf8',
                             line_terminator='\n')
    display_intermediate_time()

def test(channels_array, chunk_list_df, index_list, netcdf_dict, day, time_step,
         debug_lat, debug_lon):
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

  record = chunk_list_df[(chunk_list_df.lat_min == debug_lat_min) & (chunk_list_df.lat_max == debug_lat_max) & (chunk_list_df.lon_min == debug_lon_min) & (chunk_list_df.lon_max == debug_lon_max)]
  print(record)
  #       lat_min  lat_max  lon_min  lon_max          pred_prob  pred_cat
  #30537     10.5     18.5   -37.25   -29.25  3.056721e-10       False
  debug_id = record.index[0]

  print(f'indexes: {index_list[debug_id]}')
  #print(f'recomputed indexes: lat_min_index = {latitude_indexes[debug_lat_max]} ; lat_max_index = {latitude_indexes[debug_lat_min]} ; lon_min_index = {longitude_indexes[debug_lon_min]} ; lon_max_index = {longitude_indexes[debug_lon_max]}')

  from matplotlib import pyplot as plt
  region = channels_array[variable.value.num_id][debug_id]
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

  import extraction_utils as utils
  region = utils.extract_region(netcdf_dict[variable], variable, day, time_step, debug_lat, debug_lon)
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

  # The images must be the same, even if the image from the channels_array is based on
  # normalized values from netcdf file.
