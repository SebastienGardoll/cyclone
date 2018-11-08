#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:21:24 2018

@author: Sébastien Gardoll
"""

import os.path as path

import numpy as np
import pandas as pd

import sys

import common
from common import Era5

file_prefix = '2000_10'

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()

stats_dirname   = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{common.STATS_FILE_POSTFIX}'
stats_filename  = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{common.MERGED_CHANNEL_FILE_POSTFIX}_{common.STATS_FILE_POSTFIX}.csv'
stats_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                            stats_dirname, stats_filename)
print(f'> loading stats file: {stats_filename}')
stats_file = open(stats_file_path, 'r')
stats_dataframe = pd.read_csv(stats_file, sep=',', header=0, index_col=0,
                              na_values='')
stats_file.close()

stats_dataframe = stats_dataframe[['mean', 'stddev']]

tensor_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_TENSOR_FILE_POSTFIX}.npy'
tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, tensor_filename)

print(f'> loading tensor {tensor_filename}')
tensor = np.load(file=tensor_file_path, mmap_mode=None, allow_pickle=True)

for variable in Era5:
  num_id = variable.value.num_id
  (mean, stddev) = stats_dataframe.iloc[num_id]
  print(f'> standardizing channel {variable.name.lower()} with mean:{mean} and \
stddev: {stddev}')
