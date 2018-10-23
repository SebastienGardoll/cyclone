#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:14:50 2018

@author: Sébastien Gardoll
"""

import os.path as path

import common
from common import Era5
import extraction_utils as utils

import numpy as np

import time

class BuildTensor:

  def __init__(self, nb_images, file_prefix, file_postfix):
    self._nb_images = nb_images
    self._file_prefix = file_prefix
    self._file_postfix = file_postfix
    # Static allocation of the tensor.
    self._all_tensor = np.ndarray(shape=(nb_images, common.NB_CHANNELS,
                                        common.Y_RESOLUTION,\
                                        common.X_RESOLUTION),\
                                 dtype=np.float32)
    self._channel_tensors = dict()
    for variable in Era5:
      self._channel_tensors[variable] = np.ndarray(\
                  shape=(nb_images, common.Y_RESOLUTION, common.X_RESOLUTION),\
                  dtype=np.float32)

  def build_dataset_dict(self, year, month):
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

  def build(self, row_processor, row_iterator, has_to_skip_first_row):
    start = time.time()
    previous_year  = -1
    previous_month = -1
    nc_datasets    = None
    if has_to_skip_first_row:
      next(row_iterator) # Skip the header.
    for img_id in range(0, self._nb_images):
      row = next(row_iterator)
      print(f'processing row: {row}')
      row_tuple = row_processor(row)
      (current_year, current_month, day, time_step, lat, lon) = row_tuple
      if (current_year != previous_year) or (current_month != previous_month):
        previous_year  = current_year
        previous_month = current_month
        nc_datasets    = self.build_dataset_dict(current_year, current_month)
      for channel_index, variable in enumerate(Era5):
        region = utils.extract_region(nc_datasets[variable], variable, day,\
                                      time_step, lat, lon)
        np.copyto(dst=self._all_tensor[img_id][channel_index], src=region, casting='no')
        channel_tensor = self._channel_tensors[variable]
        np.copyto(dst=channel_tensor[img_id], src=region, casting='no')
    all_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                           f'{self._file_prefix}_all_{self._file_postfix}.npy')
    np.save(file=all_tensor_file_path, arr=self._all_tensor, allow_pickle=True)
    for variable in Era5:
      variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
       f'{self._file_prefix}_{variable.name.lower()}_{self._file_postfix}.npy')
      channel_tensor = self._channel_tensors[variable]
      np.save(file=variable_tensor_file_path, arr=channel_tensor, allow_pickle=True)
    stop = time.time()
    print("spend %f seconds processing"%((stop-start)))
    # Without channel_tensors: 1912.136137 <=> 32 mins.
    # With    channel_tensors: 1970.950752 <=> 32 mins.

