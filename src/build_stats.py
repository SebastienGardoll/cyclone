#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:45:54 2018

@author: Sébastien Gardoll
"""

import os.path as path
import os

import sys

import common
from common import Era5

import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

import time
start = time.time()

sns.set(color_codes=True)
DPI = 400
PAPER_ORIENTATION = 'portrait'
PAPER_TYPE = 'a4'
PLOT_FILE_FORMAT = 'pdf'
TRANSPARENCY = False

# Default values
file_prefix          = '2000_10'
channel_file_postfix = common.CYCLONE_CHANNEL_FILE_POSTFIX
channel_dir_path     = common.CHANNEL_PARENT_DIR_PATH
graphic_mode         = 2

if (len(sys.argv) > 4) and (sys.argv[1].strip()) and (sys.argv[2].strip()) and\
    (sys.argv[3].strip()) and (sys.argv[4].strip()):
  file_prefix          = sys.argv[1].strip()
  channel_file_postfix = sys.argv[2].strip()
  channel_dir_path     = sys.argv[3].strip()
  graphic_mode         = int(sys.argv[4].strip())

stats_parent_dir_path = path.join(channel_dir_path,
                                  f'{file_prefix}_{common.STATS_FILE_POSTFIX}')
os.makedirs(stats_parent_dir_path, exist_ok=True)

stats_list = []

channels = dict()
for variable in Era5:
  variable_channel_file_path = path.join(channel_dir_path,
             f'{file_prefix}_{variable.name.lower()}_{channel_file_postfix}.h5')
  channels[variable] = common.read_ndarray_from_hdf5(filepath=variable_channel_file_path)

for variable in Era5:
  print('')
  print(f'> computing statistics for channel {variable.name}')
  print(f'  > flatten the channel')
  channel         = channels[variable]
  raveled_channel = channel.ravel()
  if graphic_mode != 0:
    plt.clf()
    sns_plot = sns.distplot(raveled_channel, fit=stats.norm)
    plot_file_path = path.join(stats_parent_dir_path,\
      f'{variable.name.lower()}_{channel_file_postfix}_distplot.{PLOT_FILE_FORMAT}')
    plt.savefig(plot_file_path, dpi=DPI, orientation=PAPER_ORIENTATION,
                papertype=PAPER_TYPE, format=PLOT_FILE_FORMAT, transparent=TRANSPARENCY)
    if graphic_mode == 2:
      plt.show()
  mean   = raveled_channel.mean() # np.mean or std can be applied
  stddev = raveled_channel.std()  # directly on unraveled arrays.
  max_value = raveled_channel.max()
  min_value = raveled_channel.min()
  q1 = np.percentile(raveled_channel, 25)
  q2 = np.percentile(raveled_channel, 50)
  q3 = np.percentile(raveled_channel, 75)
  kurtosis_value = stats.kurtosis(raveled_channel)
  skewness_value = stats.skew(raveled_channel)
  shapiro_test   = stats.shapiro(raveled_channel)[1]
  dagostino_test = stats.normaltest(raveled_channel)[1]
  ks_test        = stats.kstest(raveled_channel, 'norm')[1]
  print(f'  > mean={mean}, stddev={stddev}, min={min_value}, max={max_value}, \
          q1={q1}, q1={q2}, q1={q3}, kurtosis={kurtosis_value}, \
          skewness={skewness_value}, shapiro-test={shapiro_test},\
          dagostino-test={dagostino_test}, ks-test={ks_test}')
  values=[variable.name.lower(), mean, stddev, min_value, max_value, q1, q2,
          q3, kurtosis_value, skewness_value, shapiro_test, dagostino_test, ks_test]
  stats_list.append(values)

stats_dataframe = pd.DataFrame(data=stats_list, columns=common.STAT_COLUMNS)

stats_dataframe_filename = f'{file_prefix}_{channel_file_postfix}_\
{common.STATS_FILE_POSTFIX}.csv'
stats_dataframe_file_path = path.join(stats_parent_dir_path,
                                      stats_dataframe_filename)
print('')
print(f'> saving {stats_dataframe_filename}')
stats_dataframe.to_csv(stats_dataframe_file_path, sep=',', na_rep='',
                       header=True, index=True, index_label='id',
                       encoding='utf8', line_terminator='\n')
stop = time.time()
formatted_time =common.display_duration((stop-start))
print('')
print(f'> spend {formatted_time} processing')
