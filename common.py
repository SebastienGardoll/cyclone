#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:22:19 2018

@author: seb
"""

                        ######## IMPORTS ########

import os.path as path

import numpy as np

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

HALF_LON_FRAME = int(X_RESOLUTION*LON_RESOLUTION / 2)
HALF_LAT_FRAME = int(Y_RESOLUTION*LAT_RESOLUTION / 2)

# Paths
DATASET_PARENT_DIR_PATH = '/home/sgardoll/ouragan/dataset'
TENSOR_PARENT_DIR_PATH  = '/home/sgardoll/ouragan/tensors'
NETCDF_PARENT_DIR_PATH  = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily'

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

                       ######## CHECKINGS ########

if X_RESOLUTION % 2 != 0:
  raise Exception("x resolution is not even")

if Y_RESOLUTION % 2 != 0:
  raise Exception("y resolution is not even")


def round_nearest(value, resolution, num_decimal):
  return round(round(value / resolution) * resolution, num_decimal)
