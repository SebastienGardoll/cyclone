#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:12:55 2018

@author: Sébastien Gardoll
"""

import csv
import os.path as path
import os

import common
from common import Era5
import extraction_utils as utils

import numpy as np

import time
start = time.time()

def build_dataset_dict(year, month):
  parent_dir_path = common.NETCDF_PARENT_DIR_PATH
  result = {Era5.MSL:utils.open_netcdf(parent_dir_path, Era5.MSL, year, month),\
            Era5.U10:utils.open_netcdf(parent_dir_path, Era5.U10, year, month),\
            Era5.V10:utils.open_netcdf(parent_dir_path, Era5.V10, year, month),\
            Era5.TCWV:utils.open_netcdf(parent_dir_path, Era5.TCWV, year, month),\
            Era5.TA200:utils.open_netcdf(parent_dir_path, Era5.TA200, year, month),\
            Era5.TA500:utils.open_netcdf(parent_dir_path, Era5.TA500, year, month),\
            Era5.U850:utils.open_netcdf(parent_dir_path, Era5.U850, year, month),\
            Era5.V850:utils.open_netcdf(parent_dir_path, Era5.V850, year, month)}
  return result

file_prefix = '2k'
cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                                 f'{file_prefix}_extraction_dataset.csv')
cyclone_db_file = open(cyclone_db_file_path, 'r')
cyclone_db_reader = csv.reader(cyclone_db_file)
nb_images   = int(os.popen(f'wc -l < {cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

# Static allocation of the tensor.
cyclone_tensor = np.ndarray(shape=(nb_images, common.NB_CHANNELS,
                                   common.Y_RESOLUTION, common.X_RESOLUTION),\
                            dtype=np.float32)
channel_tensors = dict()
for variable in Era5:
  channel_tensors[variable] = np.ndarray(shape=(nb_images, common.Y_RESOLUTION,\
                                                common.X_RESOLUTION),\
                                         dtype=np.float32)
previous_year  = -1
previous_month = -1
nc_datasets    = None
next(cyclone_db_reader) # Skip the header.
for img_id in range(0, nb_images):
  row = next(cyclone_db_reader)
  print(f'processing row: {row}')
  current_year  = int(row[3])
  current_month = int(row[4])
  day           = int(row[5])
  time_step     = int(row[6])
  lat           = float(row[8])
  lon           = float(row[9])
  if (current_year != previous_year) or (current_month != previous_month):
    previous_year  = current_year
    previous_month = current_month
    nc_datasets    = build_dataset_dict(current_year, current_month)
  for channel_index, variable in enumerate(Era5):
    region = utils.extract_region(nc_datasets[variable], variable, day,\
                                  time_step, lat, lon)
    np.copyto(dst=cyclone_tensor[img_id][channel_index], src=region, casting='no')
    channel_tensor = channel_tensors[variable]
    np.copyto(dst=channel_tensor[img_id], src=region, casting='no')

cyclone_db_file.close()

cyclone_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                                     f'{file_prefix}_all_cyclone_tensor.npy')
np.save(file=cyclone_tensor_file_path, arr=cyclone_tensor, allow_pickle=True)

for variable in Era5:
  variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                   f'{file_prefix}_{variable.name.lower()}_cyclone_tensor.npy')
  channel_tensor = channel_tensors[variable]
  np.save(file=variable_tensor_file_path, arr=channel_tensor, allow_pickle=True)

stop = time.time()
print("spend %f seconds processing"%((stop-start)))
# Without channel_tensors: 1912.136137 <=> 32 mins.
# With    channel_tensors: 1970.950752 <=> 32 mins.
