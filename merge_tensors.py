#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:58:15 2018

@author: Sébastien Gardoll
"""
import os
import os.path as path

import multiprocessing as mp
import ctypes
import sys
from multiprocessing import Pool

import csv

import common
from common import Era5

import numpy as np

from sklearn.preprocessing import StandardScaler

import time
start = time.time()

# Default value.
file_prefix   = '2000_10'
num_processes = 1

if (len(sys.argv) > 2) and (sys.argv[1].strip()) and (sys.argv[2].strip()):
  file_prefix = sys.argv[1].strip()
  num_processes = int(sys.argv[2].strip())

# It doesn't make sens to create more processes than variable to be processed.
if num_processes > len(Era5):
  num_processes = len(Era5)

os.makedirs(common.MERGED_TENSOR_PARENT_DIR_PATH, exist_ok=True)

#stats_dataframe = pd.DataFrame(columns=common.MERGED_TENSORS_STAT_COLUMNS)

# Static Allocation of the shared array of statistics.
stats_array = mp.RawArray(ctypes.ARRAY(ctypes.c_double,\
                                       len(common.MERGED_TENSORS_STAT_COLUMNS)), \
                          len(Era5))

def process_tensors(variable):
  (variable_name, variable_num_id) = variable
  print(f'> processing variable {variable_name}')

  cyclone_tensor_filename = f'{file_prefix}_{variable_name}_{common.CYCLONE_TENSOR_FILE_POSTFIX}.npy'
  print(f'> loading {cyclone_tensor_filename}')
  cyclone_variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                                                cyclone_tensor_filename)
  cyclone_var_tensor = np.load(file=cyclone_variable_tensor_file_path,\
                               mmap_mode=None, allow_pickle=True)
  no_cyclone_tensor_filename = f'{file_prefix}_{variable_name}_{common.NO_CYCLONE_TENSOR_FILE_POSTFIX}.npy'
  print(f'> loading {no_cyclone_tensor_filename}')
  no_cyclone_variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                                                   no_cyclone_tensor_filename)
  no_cyclone_var_tensor = np.load(file=no_cyclone_variable_tensor_file_path,\
                               mmap_mode=None, allow_pickle=True)

  print(f'> concatenating the tensors ({variable_name})')
  concat_var_tensor = np.concatenate((cyclone_var_tensor, no_cyclone_var_tensor))

  print(f'> standardizing the concatenated tensors ({variable_name})')
  tmp = concat_var_tensor.reshape((-1, 1), order='C')
  scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
  scaler.fit(tmp)
  mean = scaler.mean_[0]
  stddev = scaler.scale_[0]
  stats_array[variable_num_id][0] = mean
  stats_array[variable_num_id][1] = stddev
  print(f'  > mean = {mean} ({variable_name})')
  print(f'  > stddev = {stddev} ({variable_name})')
  scaler.transform(tmp)
  # TODO verify the order of the values.
  std_concat_var_tensor = tmp.reshape(concat_var_tensor.shape, order='C')
  # DEBUG
  #print(f'> [DEBUG] mean: {std_concat_var_tensor.mean()}')
  #print(f'> [DEBUG] stddev: {std_concat_var_tensor.std()}')

  std_concat_var_tensor_filename = f'{common.MERGED_TENSOR_FILE_PREFIX}_{file_prefix}_{variable_name}_{common.MERGED_TENSOR_FILE_POSTFIX}.npy'
  std_concat_var_tensor_file_path = path.join(common.MERGED_TENSOR_PARENT_DIR_PATH, \
                                              std_concat_var_tensor_filename)
  print(f'> saving {std_concat_var_tensor_filename} (shape={std_concat_var_tensor.shape})')
  np.save(file=std_concat_var_tensor_file_path, arr=std_concat_var_tensor, allow_pickle=True)

############################ STANDARDIZE CHANNELS #############################

# Parallel processing of the tensors.
list_variables = list()
for variable in Era5:
  list_variables.append((variable.name.lower(), variable.value.num_id))

# DEBUG
# process_tensors(list_variables[0])

# Giving a list of the Era5 enum items doesn't work
# (ValueError: <common.Variable object at 0x7f1448c0fc50> is not a valid Era5).
# Don't know why...
with Pool(processes = num_processes) as pool:
    pool.map(process_tensors, list_variables)

stats_filename = f'{common.MERGED_TENSOR_FILE_PREFIX}_{file_prefix}_{common.STATS_FILE_POSTFIX}.csv'
print(f'> saving stats file {stats_filename}')
stats_dataframe_file_path = path.join(common.MERGED_TENSOR_PARENT_DIR_PATH, stats_filename)
with open (stats_dataframe_file_path, 'w') as csv_file:
  csv_writter = csv.writer(csv_file, delimiter=',', lineterminator='\n')
  csv_writter.writerow(common.MERGED_TENSORS_STAT_COLUMNS)
  for line in stats_array:
    csv_writter.writerow (line)

################################ BUILD TENSOR #################################

no_cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                      f'{file_prefix}_{common.NO_CYCLONE_DB_FILE_POSTFIX}.csv')
nb_no_cyclone_images   = int(os.popen(f'wc -l < {no_cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,\
                         f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv')
nb_cyclone_images   = int(os.popen(f'wc -l < {cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

del no_cyclone_db_file_path
del cyclone_db_file_path

nb_images = nb_cyclone_images + nb_no_cyclone_images


stop = time.time()
print("> spend %f seconds processing"%((stop-start)))

''' DEBUG
a =   cyclone_var_tensor = np.load(file='/home/seb/private/home_ciclad/ouragan/merged_tensors/merged_2000_10_msl_tensor.npy',\
                               mmap_mode=None, allow_pickle=True)

b =   cyclone_var_tensor = np.load(file='/home/seb/private/home_ciclad/ouragan/merged_tensors/merged_2000_10_ta200_tensor.npy',\
                               mmap_mode=None, allow_pickle=True)

c =   cyclone_var_tensor = np.load(file='/home/seb/private/home_ciclad/ouragan/merged_tensors/merged_2000_10_ta500_tensor.npy',\
                               mmap_mode=None, allow_pickle=True)


d = np.stack((a, b, c), axis=3)
print(d.shape)
print('------------')
print(a[0][0][0])
print(b[0][0][0])
print(c[0][0][0])
print(d[0][0][0])
print('------------')
print(a[0][31][31])
print(b[0][31][31])
print(c[0][31][31])
print(d[0][31][31])
print('------------')
print(a[0][31][0])
print(b[0][31][0])
print(c[0][31][0])
print(d[0][31][0])
print('------------')
print(a[0][31][31])
print(b[0][31][31])
print(c[0][31][31])
print(d[0][31][31])
print('------------')
print(a[146][0][0])
print(b[146][0][0])
print(c[146][0][0])
print(d[146][0][0])
print('------------')
print(a[146][31][31])
print(b[146][31][31])
print(c[146][31][31])
print(d[146][31][31])
print('------------')
print(a[146][31][0])
print(b[146][31][0])
print(c[146][31][0])
print(d[146][31][0])
print('------------')
print(a[146][31][31])
print(b[146][31][31])
print(c[146][31][31])
print(d[146][31][31])
'''
