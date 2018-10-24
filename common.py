#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:22:19 2018

@author: Sébastien Gardoll
"""

                        ######## IMPORTS ########

import os.path as path
import os

import numpy as np

from datetime import datetime
from datetime import timedelta

                    ######## STATIC VARIABLES ########

# NetCDF resolution.
TIME_SAMPLING = 4
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

# Paths
NETCDF_PARENT_DIR_PATH          = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily'
ROOT_DIR_PARENT                 = '/home/sgardoll/ouragan'
DATASET_PARENT_DIR_PATH         = path.join(ROOT_DIR_PARENT, 'dataset')
TENSOR_PARENT_DIR_PATH          = path.join(ROOT_DIR_PARENT, 'tensors')
MERGED_TENSOR_PARENT_DIR_PATH   = path.join(ROOT_DIR_PARENT, 'merged_tensors')
SHUFFLED_TENSOR_PARENT_DIR_PATH = path.join(ROOT_DIR_PARENT, 'shuffled_tensors')


ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)

CYCLONE_TENSOR_FILE_POSTFIX    = 'cyclone_tensor'
CYCLONE_DB_FILE_POSTFIX        = 'extraction_dataset'
NO_CYCLONE_TENSOR_FILE_POSTFIX = 'no_cyclone_tensor'
NO_CYCLONE_DB_FILE_POSTFIX     = 'no_cyclone_dataset'
MERGED_TENSOR_FILE_POSTFIX     = 'tensor'

STAT_COLUMNS = ['variable', 'mean', 'stddev', 'min', 'max', 'q1', 'q2',\
                'q3','kurtosis', 'skewness', 'shapiro-test', 'dagostino-test',\
                'ks-test']

                       ######## STATIC CLASSES ########

class Variable:

  def __init__(self, num_id, str_id, level = None, index_mapping = None):
    self.num_id = num_id
    self.str_id = str_id
    self.level = level
    self.index_mapping = index_mapping

# ERA5 variable names.
from enum import Enum

class Era5 (Enum):
  MSL   = Variable(0, 'msl')
  TCWV  = Variable(1, 'tcwv')
  V10   = Variable(2, 'v10')
  U10   = Variable(3, 'u10')
  TA200 = Variable(4, 'ta', 200,\
    np.load(path.join(DATASET_PARENT_DIR_PATH,'ta_indexes.npy')).item())
  TA500 = Variable(5, 'ta', 500,\
    np.load(path.join(DATASET_PARENT_DIR_PATH,'ta_indexes.npy')).item())
  U850  = Variable(6, 'u', 850,\
    np.load(path.join(DATASET_PARENT_DIR_PATH, 'u_indexes.npy')).item())
  V850  = Variable(7, 'v', 850,\
    np.load(path.join(DATASET_PARENT_DIR_PATH, 'v_indexes.npy')).item())

NB_CHANNELS = len(Era5)

                       ######## FUNCTIONS ########

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

def round_nearest(value, resolution, num_decimal):
  return round(round(value / resolution) * resolution, num_decimal)

                       ######## CHECKINGS ########

if X_RESOLUTION % 2 != 0:
  raise Exception("x resolution is not even")

if Y_RESOLUTION % 2 != 0:
  raise Exception("y resolution is not even")


os.makedirs(TENSOR_PARENT_DIR_PATH, exist_ok=True)
os.makedirs(MERGED_TENSOR_PARENT_DIR_PATH, exist_ok=True)
os.makedirs(SHUFFLED_TENSOR_PARENT_DIR_PATH, exist_ok=True)