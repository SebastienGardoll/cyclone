#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:45:54 2018

@author: seb
"""

import os.path as path
import os
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

CYCLONE_STAT_COLUMNS = ['variable', 'mean', 'stddev', 'min', 'max', 'q1', 'q2',\
                        'q3','kurtosis', 'skewness', 'shapiro-test', 'dagostino-test',\
                        'ks-test']

file_prefix  = '2000_10'
file_postfix = 'cyclone_tensor.npy'

cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                                 f'{file_prefix}_extraction_dataset.csv')
nb_images   = int(os.popen(f'wc -l < {cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

stats_dataframe = pd.DataFrame(columns=CYCLONE_STAT_COLUMNS)

channel_tensors = dict()
for variable in Era5:
  variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                       f'{file_prefix}_{variable.name.lower()}_{file_postfix}')
  channel_tensors[variable] = np.load(file=variable_tensor_file_path,\
                                      mmap_mode=None, allow_pickle=True)
for variable in Era5:
  print('')
  print(f'> computing statistics for {variable.name} tensor')
  print(f'  > flatten the tensor')
  channel_tensor         = channel_tensors[variable]
  raveled_channel_tensor = channel_tensor.ravel()
  sns.distplot(raveled_channel_tensor, fit=stats.norm)
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
          dagostino-test={dagostino_test}, ks-test={ks_test}')
  values=[variable.name.lower(), mean, stddev, min_value, max_value, q1, q2,\
          q3, kurtosis_value, skewness_value, shapiro_test, dagostino_test, ks_test]
  stats_row = pd.Series(values, index=CYCLONE_STAT_COLUMNS)
  stats_dataframe = stats_dataframe.append(stats_row, ignore_index=True)

stats_dataframe_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                                      f'{file_prefix}_tensor_stats.csv')
stats_dataframe.to_csv(stats_dataframe_file_path, sep = ',', na_rep = '',\
                       header = True, index = True, index_label='id',\
                       encoding = 'utf8', line_terminator = '\n')

stop = time.time()
print('')
print("> spend %f seconds processing"%((stop-start)))