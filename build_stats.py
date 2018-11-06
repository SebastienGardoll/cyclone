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
file_prefix         = '2000_10'
tensor_file_postfix = common.CYCLONE_TENSOR_FILE_POSTFIX
tensor_dir_path     = common.TENSOR_PARENT_DIR_PATH
graphic_mode        = 2

if (len(sys.argv) > 4) and (sys.argv[1].strip()) and (sys.argv[2].strip()) and\
    (sys.argv[3].strip()) and (sys.argv[4].strip()):
  file_prefix         = sys.argv[1].strip()
  tensor_file_postfix = sys.argv[2].strip()
  tensor_dir_path     = sys.argv[3].strip()
  graphic_mode        = int(sys.argv[4].strip())

stats_parent_dir_path = path.join(tensor_dir_path,\
                                  f'{file_prefix}_stats')
os.makedirs(stats_parent_dir_path, exist_ok=True)

stats_dataframe = pd.DataFrame(columns=common.STAT_COLUMNS)

channel_tensors = dict()
for variable in Era5:
  variable_tensor_file_path = path.join(tensor_dir_path,\
                       f'{file_prefix}_{variable.name.lower()}_{tensor_file_postfix}.npy')
  channel_tensors[variable] = np.load(file=variable_tensor_file_path,\
                                      mmap_mode=None, allow_pickle=True)
for variable in Era5:
  print('', flush=True)
  print(f'> computing statistics for {variable.name} tensor', flush=True)
  print(f'  > flatten the tensor', flush=True)
  channel_tensor         = channel_tensors[variable]
  raveled_channel_tensor = channel_tensor.ravel()
  if graphic_mode != 0:
    sns.distplot(raveled_channel_tensor, fit=stats.norm)
    plot_file_path = path.join(stats_parent_dir_path,\
      f'{variable.name.lower()}_{tensor_file_postfix}_distplot.{PLOT_FILE_FORMAT}')
    plt.savefig(plot_file_path, dpi=DPI, orientation = PAPER_ORIENTATION,\
                papertype = PAPER_TYPE, format = PLOT_FILE_FORMAT,\
                transparent = TRANSPARENCY)
    if graphic_mode == 2:
      plt.show()
  mean   = raveled_channel_tensor.mean()
  stddev = raveled_channel_tensor.std()
  max_value = raveled_channel_tensor.max()
  min_value = raveled_channel_tensor.min()
  q1 = np.percentile(raveled_channel_tensor, 25)
  q2 = np.percentile(raveled_channel_tensor, 50)
  q3 = np.percentile(raveled_channel_tensor, 75)
  kurtosis_value = stats.kurtosis(raveled_channel_tensor)
  skewness_value = stats.skew(raveled_channel_tensor)
  shapiro_test   = stats.shapiro(raveled_channel_tensor)[1]
  dagostino_test = stats.normaltest(raveled_channel_tensor)[1]
  ks_test        = stats.kstest(raveled_channel_tensor, 'norm')[1]
  print(f'  > mean={mean}, stddev={stddev}, min={min_value}, max={max_value}, \
          q1={q1}, q1={q2}, q1={q3}, kurtosis={kurtosis_value}, \
          skewness={skewness_value}, shapiro-test={shapiro_test},\
          dagostino-test={dagostino_test}, ks-test={ks_test}', flush=True)
  values=[variable.name.lower(), mean, stddev, min_value, max_value, q1, q2,\
          q3, kurtosis_value, skewness_value, shapiro_test, dagostino_test, ks_test]
  stats_row = pd.Series(values, index=common.STAT_COLUMNS)
  stats_dataframe = stats_dataframe.append(stats_row, ignore_index=True)

stats_dataframe_filename = f'{file_prefix}_{tensor_file_postfix}_stats.csv'
stats_dataframe_file_path = path.join(stats_parent_dir_path,\
                                      stats_dataframe_filename)
print('', flush=True)
print(f'> saving {stats_dataframe_filename}', flush=True)
stats_dataframe.to_csv(stats_dataframe_file_path, sep = ',', na_rep = '',\
                       header = True, index = True, index_label='id',\
                       encoding = 'utf8', line_terminator = '\n')
stop = time.time()
print('', flush=True)
print("> spend %f seconds processing"%((stop-start)), flush=True)