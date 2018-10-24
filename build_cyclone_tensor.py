#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:05:33 2018

@author: Sébastien Gardoll
"""
import csv
import os.path as path
import os

import sys

import common

from build_tensor import BuildTensor

def row_processor(row):
  year  = int(row[3])
  month = int(row[4])
  day           = int(row[5])
  time_step     = int(row[6])
  lat           = float(row[8])
  lon           = float(row[9])
  return (year, month, day, time_step, lat, lon)

# Default values.
file_prefix = '2000_10'

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()

file_postfix = common.CYCLONE_TENSOR_FILE_POSTFIX

cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                         f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv')
cyclone_db_file = open(cyclone_db_file_path, 'r')
cyclone_db_reader = csv.reader(cyclone_db_file)
nb_images   = int(os.popen(f'wc -l < {cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

bt = BuildTensor(nb_images, file_prefix, file_postfix)

bt.build(row_processor, cyclone_db_reader, True)

cyclone_db_file.close()