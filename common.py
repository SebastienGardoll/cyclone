#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:22:19 2018

@author: Sébastien Gardoll
"""

                        ######## IMPORTS ########

import os.path as path
import os

import csv

from enum import Enum

import numpy as np
import h5py

from datetime import datetime
from datetime import timedelta

                    ######## STATIC VARIABLES ########

# System
MEGA_BYTES_FACTOR = 1024*1024
SUCCESS_CODE = 0
ERROR_CODE   = 1
CANCEL_CODE  = 2

# NetCDF resolution.
_4XDAILY_TIME_SAMPLING = 4
HOURLY_TIME_SAMPLING   = 24
LAT_RESOLUTION = 0.25
LON_RESOLUTION = 0.25

NUM_DECIMAL_LAT = 2
NUM_DECIMAL_LON = 2

# Tensor resolution. Must be even numbers.
X_RESOLUTION = 32
Y_RESOLUTION = 32

HALF_LAT_FRAME = int(Y_RESOLUTION*LAT_RESOLUTION / 2)
HALF_LON_FRAME = int(X_RESOLUTION*LON_RESOLUTION / 2)

LAT_FRAME = HALF_LAT_FRAME * 2
LON_FRAME = HALF_LON_FRAME * 2

# 0 refers to a no cyclone.
# 1 refers to a cyclone.
NUM_CLASSES      = 2
CYCLONE_LABEL    = 1.0
NO_CYCLONE_LABEL = 0.0
CYCLONE_CAT      = 'cyclone'
NO_CYCLONE_CAT   = 'no cyclone'

# Paths
NETCDF_PARENT_DIR_PATH         = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily'
ROOT_DATA_DIR_PATH             = '/data/sgardoll/cyclone_data'
SCRIPT_DIR_PATH                = '/home/sgardoll/cyclone/src'
DATASET_PARENT_DIR_PATH        = path.join(ROOT_DATA_DIR_PATH, 'dataset')
CHANNEL_PARENT_DIR_PATH        = path.join(ROOT_DATA_DIR_PATH, 'channels')
MERGED_CHANNEL_PARENT_DIR_PATH = path.join(ROOT_DATA_DIR_PATH, 'merged_channels')
TENSOR_PARENT_DIR_PATH         = path.join(ROOT_DATA_DIR_PATH, 'tensor')
CNN_PARENT_DIR_PATH            = path.join(ROOT_DATA_DIR_PATH, 'cnn')
PREDICT_TENSOR_PARENT_DIR_PATH = path.join(ROOT_DATA_DIR_PATH, 'predict_tensor')

CYCLONE_ALL_DB_FILE_PATH = path.join(DATASET_PARENT_DIR_PATH,
                                     'all_cyclone_dataset.csv')

STAT_SCRIPT_NAME               = 'build_stats.py'

ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)

CYCLONE_CHANNEL_FILE_POSTFIX    = 'cyclone_channel'
CYCLONE_DB_FILE_POSTFIX         = 'cyclone_dataset'
NO_CYCLONE_CHANNEL_FILE_POSTFIX = 'no_cyclone_channel'
NO_CYCLONE_DB_FILE_POSTFIX      = 'no_cyclone_dataset'
MERGED_CHANNEL_FILE_POSTFIX     = 'channel'
SHUFFLED_TENSOR_FILE_POSTFIX    = 'tensor'
SHUFFLED_LABELS_FILE_POSTFIX    = 'labels'
STATS_FILE_POSTFIX              = 'stats'
CNN_FILE_POSTFIX                = 'model'
PREDICTION_FILE_POSTFIX         = 'predict'

MERGED_CHANNEL_FILE_PREFIX      = 'merged'
SHUFFLED_FILE_PREFIX            = 'shuffled'


DATA_PARENT_DIR_PATH                  = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025'
ONE_LEVEL_DATA_FILE_PATH_PREFIX       = f'{DATA_PARENT_DIR_PATH}/hourly/AN_SF'
ONE_LEVEL_DATA_FILE_NAME_POSTFIX      = 'as1e5.GLOBAL_025.nc'
MULTIPLE_LEVEL_DATA_FILE_PATH_PREFIX  = f'{DATA_PARENT_DIR_PATH}/4xdaily/AN_PL'
MULTIPLE_LEVEL_DATA_FILE_NAME_POSTFIX = 'aphe5.GLOBAL_025.nc'

STAT_COLUMNS = ['variable', 'mean', 'stddev', 'min', 'max', 'q1', 'q2',
                'q3','kurtosis', 'skewness', 'shapiro-test', 'dagostino-test',
                'ks-test']

MERGED_CHANNEL_STAT_COLUMNS = ['mean', 'stddev']


                       ######## FUNCTIONS ########

def write_ndarray_to_hdf5(filepath, ndarray):
  hdf5_file = h5py.File(filepath, 'w')
  hdf5_file.create_dataset('dataset', data=ndarray)
  hdf5_file.close()

def read_ndarray_from_hdf5(filepath):
  hdf5_file = h5py.File(filepath, 'r')
  data = hdf5_file.get('dataset')
  return np.array(data)

def write_dict_to_csv(filepath, dictionary):
  with open (filepath, 'w') as csv_file:
    csv_writter = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    for key, value in dictionary.items():
      csv_writter.writerow([key, value])

def read_dict_from_csv(filepath, cast_key, cast_value):
  result = dict()
  with open (filepath, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', lineterminator='\n')
    for row in csv_reader:
      result[cast_key(row[0])]=cast_value(row[1])
  return result

def subtract_one_day(year, month, day):
  date = datetime(year=year, month=month, day=day)
  return _subtract_one_day(date)


def subtract_delta(year, month, day, delta):
  result = datetime(year=year, month=month, day=day) - delta
  return result


def _subtract_one_day(date):
  result = date - ONE_DAY
  return result


def is_overlapping(lat1, lon1, lat2, lon2):
  if abs(lat1-lat2) <= LAT_FRAME:
    if abs(lon1 -lon2) <= LON_FRAME:
      return True
    else:
      return False
  else:
      return False

def to_cat(label_value):
  return CYCLONE_CAT if label_value == CYCLONE_LABEL else NO_CYCLONE_CAT

def round_nearest(value, resolution, num_decimal):
  return round(round(value / resolution) * resolution, num_decimal)

def format_pourcentage(value):
  return round(value*100, 2)

def format_float_number(float_value):
  if float_value < 0.001 :
    return f'{float_value:.2e}'
  else:
    return f'{float_value:.3f}'

def display_duration(time_in_sec):
  remainder = time_in_sec % 60
  if remainder == time_in_sec:
    return f'{time_in_sec:.2f} seconds'
  else:
    seconds = remainder
    minutes = int(time_in_sec / 60)
    remainder = minutes % 60
    if remainder == minutes:
      return f'{minutes} mins, {seconds:.2f} seconds'
    else:
      hours   = int(minutes / 60)
      minutes = remainder
      remainder = hours % 24
      if remainder == hours:
        return f'{hours} hours, {minutes} mins, {seconds:.2f} seconds'
      else:
        days = int(hours / 24)
        hours = remainder
        return f'{days} days, {hours} hours, {minutes} mins, {seconds:.2f} seconds'


                       ######## STATIC CLASSES ########


class Variable:

  def __init__(self, num_id, str_id, level = None, index_mapping = None):
    self.num_id = num_id
    self.str_id = str_id
    self.level = level
    self.index_mapping = index_mapping
    if level:
      self.file_path_prefix = MULTIPLE_LEVEL_DATA_FILE_PATH_PREFIX
      self.filename_postfix = MULTIPLE_LEVEL_DATA_FILE_NAME_POSTFIX
    else:
      self.file_path_prefix = ONE_LEVEL_DATA_FILE_PATH_PREFIX
      self.filename_postfix = ONE_LEVEL_DATA_FILE_NAME_POSTFIX

  def compute_file_path(self, year, month):
    return f'{self.file_path_prefix}/{year}/{self.str_id}.{year}{month:02d}.{self.filename_postfix}'

  def is_hourly(self):
    return self.level == None

  def compute_time_index(self, num_day, time_step = 0):
    if self.is_hourly():
      return Variable._compute_time_index_hourly(num_day, time_step)
    else:
      return Variable._compute_time_index_4xdaily(num_day, time_step)

  @staticmethod
  def _compute_time_index_4xdaily(num_day, time_step = 0):
    # Handle over spec time_step.
    if time_step >= _4XDAILY_TIME_SAMPLING:
      days_to_add = int(time_step / _4XDAILY_TIME_SAMPLING)
      time_step = time_step % _4XDAILY_TIME_SAMPLING
      num_day = num_day + days_to_add
    return _4XDAILY_TIME_SAMPLING*(num_day-1) + time_step

  @staticmethod
  # Convert 4xdaily basis num_day and time_step into hourly basis index of time.
  def _compute_time_index_hourly(num_day, time_step = 0):
    # Handle over spec time_step.
    if time_step >= HOURLY_TIME_SAMPLING:
      days_to_add = int(time_step / _4XDAILY_TIME_SAMPLING)
      time_step = time_step % _4XDAILY_TIME_SAMPLING
      num_day = num_day + days_to_add
    return HOURLY_TIME_SAMPLING*(num_day-1) + time_step

# ERA5 variable names.
class Era5 (Enum):
  MSL   = Variable(0, 'msl')
  TCWV  = Variable(1, 'tcwv')
  V10   = Variable(2, 'v10')
  U10   = Variable(3, 'u10')
  TA200 = Variable(4, 'ta', 200,
    read_dict_from_csv(path.join(DATASET_PARENT_DIR_PATH,'ta_indexes.csv'), int, int))
  TA500 = Variable(5, 'ta', 500,
   read_dict_from_csv(path.join(DATASET_PARENT_DIR_PATH,'ta_indexes.csv'), int, int))
  U850  = Variable(6, 'u', 850,
    read_dict_from_csv(path.join(DATASET_PARENT_DIR_PATH, 'u_indexes.csv'), int, int))
  V850  = Variable(7, 'v', 850,
    read_dict_from_csv(path.join(DATASET_PARENT_DIR_PATH, 'v_indexes.csv'), int, int))


NB_CHANNELS = len(Era5)

                       ######## CHECKINGS ########


if X_RESOLUTION % 2 != 0:
  raise Exception("x resolution is not even")

if Y_RESOLUTION % 2 != 0:
  raise Exception("y resolution is not even")


os.makedirs(CHANNEL_PARENT_DIR_PATH, exist_ok=True)
os.makedirs(MERGED_CHANNEL_PARENT_DIR_PATH, exist_ok=True)
os.makedirs(TENSOR_PARENT_DIR_PATH, exist_ok=True)
os.makedirs(CNN_PARENT_DIR_PATH, exist_ok=True)
os.makedirs(PREDICT_TENSOR_PARENT_DIR_PATH, exist_ok=True)
