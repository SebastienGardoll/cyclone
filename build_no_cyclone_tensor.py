#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:32:12 2018

@author: Sébastien Gardoll
"""

import csv
import os.path as path
import os

import sys

import common

from build_tensor import BuildTensor

def row_processor(row):
  year  = int(row[1])
  month = int(row[2])
  day           = int(row[3])
  time_step     = int(row[4])
  lat           = float(row[5])
  lon           = float(row[6])
  return (year, month, day, time_step, lat, lon)

# Default value.
file_prefix = '2000_10'
has_to_show_plot = True

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()
  has_to_show_plot = False

file_postfix = common.NO_CYCLONE_TENSOR_FILE_POSTFIX

no_cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                      f'{file_prefix}_{common.NO_CYCLONE_DB_FILE_POSTFIX}.csv')
no_cyclone_db_file = open(no_cyclone_db_file_path, 'r')
no_cyclone_db_reader = csv.reader(no_cyclone_db_file)
nb_images   = int(os.popen(f'wc -l < {no_cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

bt = BuildTensor(nb_images, file_prefix, file_postfix)

bt.build(row_processor, no_cyclone_db_reader, True, has_to_show_plot)

no_cyclone_db_file.close()