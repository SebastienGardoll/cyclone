#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:14:50 2018

@author: Sébastien Gardoll
"""

import psutil
import os
import os.path as path

import common
from common import Era5
import extraction_utils as utils

import time

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


class BuildChannels:

  def __init__(self, nb_images, file_prefix, file_postfix):
    self._nb_images = nb_images
    self._file_prefix = file_prefix
    self._file_postfix = file_postfix
    self._channels = dict()
    for variable in Era5:
      self._channels[variable] = np.ndarray(
                  shape=(nb_images, common.Y_RESOLUTION, common.X_RESOLUTION),
                  dtype=np.float32)

  @staticmethod
  def close_dataset_dict(dataset_dict):
    for dataset in dataset_dict.values():
      dataset.close()

  def build(self, row_processor, row_iterator, has_to_skip_first_row,
            has_to_show_plot):
    start = time.time()
    previous_year  = -1
    previous_month = -1
    nc_datasets    = None
    if has_to_skip_first_row:
      next(row_iterator) # Skip the header.
    for img_id in range(0, self._nb_images):
      row = next(row_iterator)
      print(f'> processing row: {row}')
      row_tuple = row_processor(row)
      (current_year, current_month, day, time_step, lat, lon) = row_tuple
      if (current_year != previous_year) or (current_month != previous_month):
        previous_year  = current_year
        previous_month = current_month
        if nc_datasets:
          self.close_dataset_dict(nc_datasets)
        nc_datasets    = utils.build_dataset_dict(current_year, current_month)
      for channel_index, variable in enumerate(Era5):
        region = utils.extract_region(nc_datasets[variable], variable, day,
                                      time_step, lat, lon)
        channel = self._channels[variable]
        np.copyto(dst=channel[img_id], src=region, casting='no')
        if has_to_show_plot:
          print(f'\n  > showing variable {variable.name.lower()}')
          plt.imshow(region, cmap='gist_rainbow_r',interpolation="none")
          plt.show()

    for variable in Era5:
      variable_channel_file_path = path.join(common.CHANNEL_PARENT_DIR_PATH,
        f'{self._file_prefix}_{variable.name.lower()}_{self._file_postfix}.npy')
      channel = self._channels[variable]
      print(f'> saving {variable.name.lower()} channel (shape={channel.shape})')
      np.save(file=variable_channel_file_path, arr=channel, allow_pickle=True)
    stop = time.time()
    formatted_time =common.display_duration((stop-start))
    print(f'> spend {formatted_time} processing')
    # 1912.136137 <=> 32 mins.
    process = psutil.Process(os.getpid())
    print(f'> maximum memory footprint: {process.memory_info().rss/common.MEGA_BYTES_FACTOR:.2f} MiB')
