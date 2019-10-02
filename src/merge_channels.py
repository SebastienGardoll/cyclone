#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:58:15 2018

@author: Sébastien Gardoll
"""
import psutil

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

# Static Allocation of the shared array of statistics.
stats_array = mp.RawArray(ctypes.ARRAY(ctypes.c_double,
                          len(common.MERGED_CHANNEL_STAT_COLUMNS)),
                          len(Era5))


def process_channels(variable):
  (variable_name, variable_num_id) = variable
  print(f'> processing variable {variable_name}')

  cyclone_channel_filename = f'{file_prefix}_{variable_name}_\
{common.CYCLONE_CHANNEL_FILE_POSTFIX}.h5'
  print(f'> loading {cyclone_channel_filename}')
  cyclone_channel_file_path = path.join(common.CHANNEL_PARENT_DIR_PATH,
                                                cyclone_channel_filename)
  cyclone_channel = common.read_ndarray_from_hdf5(filepath=cyclone_channel_file_path)
  no_cyclone_channel_filename = f'{file_prefix}_{variable_name}_\
{common.NO_CYCLONE_CHANNEL_FILE_POSTFIX}.h5'
  print(f'> loading {no_cyclone_channel_filename}')
  no_cyclone_channel_file_path = path.join(common.CHANNEL_PARENT_DIR_PATH,
                                           no_cyclone_channel_filename)
  no_cyclone_channel = common.read_ndarray_from_hdf5(filepath=no_cyclone_channel_file_path)

  print(f'> concatenating the channels ({variable_name})')
  concat_channels = np.concatenate((cyclone_channel, no_cyclone_channel))

  print(f'> standardizing the concatenated channels ({variable_name})')
  tmp = concat_channels.reshape((-1, 1), order='C')
  scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
  scaler.fit(tmp)
  mean = scaler.mean_[0]
  stddev = scaler.scale_[0]
  stats_array[variable_num_id][0] = mean
  stats_array[variable_num_id][1] = stddev
  print(f'  > mean = {mean} ({variable_name})')
  print(f'  > stddev = {stddev} ({variable_name})')
  scaler.transform(tmp)
  std_concat_channels = tmp.reshape(concat_channels.shape, order='C')
  # DEBUG
  # print(f'> [DEBUG] mean: {std_concat_channels.mean()}')
  # print(f'> [DEBUG] stddev: {std_concat_channels.std()}')

  std_concat_channels_filename = f'{common.MERGED_CHANNEL_FILE_PREFIX}_\
{file_prefix}_{variable_name}_{common.MERGED_CHANNEL_FILE_POSTFIX}.h5'
  std_concat_channels_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                                              std_concat_channels_filename)
  print(f'> saving {std_concat_channels_filename} (shape={std_concat_channels.shape})')
  common.write_ndarray_to_hdf5(filepath=std_concat_channels_file_path,  ndarray=std_concat_channels)

############################ STANDARDIZE CHANNELS #############################


# Parallel processing of the channels.
list_variables = list()
for variable in Era5:
  list_variables.append((variable.name.lower(), variable.value.num_id))

# DEBUG
# process_channels(list_variables[0])

# Giving a list of the Era5 enum items doesn't work
# (ValueError: <common.Variable object at 0x7f1448c0fc50> is not a valid Era5).
# Don't know why...
print(f'> allocating {num_processes} process(es)')
with Pool(processes = num_processes) as pool:
    pool.map(process_channels, list_variables)

stats_filename = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{common.STATS_FILE_POSTFIX}.csv'
print(f'> saving stats file {stats_filename}')
stats_dataframe_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                                      stats_filename)
with open (stats_dataframe_file_path, 'w') as csv_file:
  csv_writter = csv.writer(csv_file, delimiter=',', lineterminator='\n')
  csv_writter.writerow(common.MERGED_CHANNEL_STAT_COLUMNS)
  for line in stats_array:
    csv_writter.writerow (line)

################################ BUILD TENSOR #################################

print('> building the tensor')

cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,
                          f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv')
nb_cyclone_images   = int(os.popen(f'wc -l < {cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

no_cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,
                       f'{file_prefix}_{common.NO_CYCLONE_DB_FILE_POSTFIX}.csv')
nb_no_cyclone_images   = int(os.popen(f'wc -l < {no_cyclone_db_file_path}').read()[:-1])-1 # -1 <=> header.

del no_cyclone_db_file_path
del cyclone_db_file_path

nb_images = nb_cyclone_images + nb_no_cyclone_images

# Insertion order is supported.
channels = []

for variable in Era5:
  channel_filename = f'{common.MERGED_CHANNEL_FILE_PREFIX}_{file_prefix}_\
{variable.name.lower()}_{common.MERGED_CHANNEL_FILE_POSTFIX}.h5'
  print(f'> loading {channel_filename}')
  channel_file_path = path.join(common.MERGED_CHANNEL_PARENT_DIR_PATH,
                                channel_filename)
  channel_imgs = common.read_ndarray_from_hdf5(filepath=channel_file_path)
  channels.append(channel_imgs)

print('> stacking the channels')
tensor = np.stack(channels, axis=3)

# DEBUG
# tmp = np.ravel(tensor)
# print(tmp.mean())
# print(tmp.std())

print(f'> building the labels (cyclones: {nb_cyclone_images} ; \
no cyclones: {nb_no_cyclone_images})')

labels_cyclone = np.ndarray(shape=(nb_cyclone_images), dtype=np.float32)
labels_cyclone.fill(common.CYCLONE_LABEL)

labels_no_cyclone = np.ndarray(shape=(nb_no_cyclone_images), dtype=np.float32)
labels_no_cyclone.fill(common.NO_CYCLONE_LABEL)

merge_labels = np.concatenate((labels_cyclone, labels_no_cyclone))

print('> shuffling the tensor and the labels')
permutation = np.random.permutation((nb_cyclone_images + nb_no_cyclone_images))
shuffled_tensor = tensor[permutation]
process = psutil.Process(os.getpid())
max_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
del tensor
shuffled_labels = merge_labels[permutation]
del merge_labels

tensor_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_TENSOR_FILE_POSTFIX}.h5'
print(f'> saving the tensor {tensor_filename} (shape: {shuffled_tensor.shape})')
tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, tensor_filename)
common.write_ndarray_to_hdf5(filepath=tensor_file_path,  ndarray=shuffled_tensor)

shuffled_labels_filename  = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
print(f'> saving the labels {shuffled_labels_filename}')
shuffled_labels_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,
                                      shuffled_labels_filename)
common.write_ndarray_to_hdf5(filepath=shuffled_labels_file_path,  ndarray=shuffled_labels)

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')
process = psutil.Process(os.getpid())
current_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
if current_mem > max_mem:
  max_mem = current_mem
print(f'> maximum memory footprint: {max_mem:.2f} MiB')

################################### DEBUG ######################################

''' DEBUG
a = common.read_ndarray_from_hdf5(filepath='/home/seb/private/home_ciclad/cyclone/channels/2000_10_msl_cyclone_channel.h5')

print(a.shape)

b = common.read_ndarray_from_hdf5(filepath='/home/seb/private/home_ciclad/cyclone/channels/2000_10_msl_no_cyclone_channel.h5')

print(b.shape)

c = np.concatenate((a, b))

tmp = c.reshape((-1, 1), order='C')
scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
scaler.fit(tmp)
mean = scaler.mean_[0]
stddev = scaler.scale_[0]
print(f'  > mean = {mean}')
print(f'  > stddev = {stddev}')
scaler.transform(tmp)
d = tmp.reshape(c.shape, order='C')

def compute_std(value, mean, stddev):
  return ((value - mean)/stddev)

print(d.shape)
print('------------')
print(compute_std(a[0][0][0], mean, stddev))
print(compute_std(b[0][0][0], mean, stddev))
print(d[0][0][0])
print(d[a.shape[0]][0][0])
print('------------')
print(compute_std(a[0][0][31], mean, stddev))
print(compute_std(b[0][0][31], mean, stddev))
print(d[0][0][31])
print(d[a.shape[0]][0][31])
print('------------')
print(compute_std(a[0][31][0], mean, stddev))
print(compute_std(b[0][31][0], mean, stddev))
print(d[0][31][0])
print(d[a.shape[0]][31][0])
print('------------')
print(compute_std(a[0][31][31], mean, stddev))
print(compute_std(b[0][31][31], mean, stddev))
print(d[0][31][31])
print(d[a.shape[0]][31][31])
print('------------')
print(compute_std(a[(a.shape[0]-1)][0][0], mean, stddev))
print(compute_std(b[(b.shape[0]-1)][0][0], mean, stddev))
print(d[(a.shape[0]-1)][0][0])
print(d[(d.shape[0]-1)][0][0])
print('------------')
print(compute_std(a[(a.shape[0]-1)][0][31], mean, stddev))
print(compute_std(b[(b.shape[0]-1)][0][31], mean, stddev))
print(d[(a.shape[0]-1)][0][31])
print(d[(d.shape[0]-1)][0][31])
print('------------')
print(compute_std(a[(a.shape[0]-1)][31][0], mean, stddev))
print(compute_std(b[(b.shape[0]-1)][31][0], mean, stddev))
print(d[(a.shape[0]-1)][31][0])
print(d[(d.shape[0]-1)][31][0])
print('------------')
print(compute_std(a[(a.shape[0]-1)][31][31], mean, stddev))
print(compute_std(b[(b.shape[0]-1)][31][31], mean, stddev))
print(d[(a.shape[0]-1)][31][31])
print(d[(d.shape[0]-1)][31][31])

a =   cyclone_var_tensor = common.read_ndarray_from_hdf5(filepath='/home/seb/private/home_ciclad/cyclone/merged_channels/merged_2000_10_msl_channel.h5')

b =   cyclone_var_tensor = common.read_ndarray_from_hdf5(filepath='/home/seb/private/home_ciclad/cyclone/merged_channels/merged_2000_10_ta200_channel.h5')

c =   cyclone_var_tensor = common.read_ndarray_from_hdf5(filepath='/home/seb/private/home_ciclad/cyclone/merged_channels/merged_2000_10_ta500_channel.h5')


d = np.stack((a, b, c), axis=3)
print(d.shape)
print('------------')
print(a[0][0][0])
print(b[0][0][0])
print(c[0][0][0])
print(d[0][0][0])
print('------------')
print(a[0][0][31])
print(b[0][0][31])
print(c[0][0][31])
print(d[0][0][31])
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
print(a[146][0][31])
print(b[146][0][31])
print(c[146][0][31])
print(d[146][0][31])
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
