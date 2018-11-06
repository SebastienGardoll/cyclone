#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:58:15 2018

@author: Sébastien Gardoll
"""
import os.path as path

import sys

import common
from common import Era5

import numpy as np

import subprocess

import time
start = time.time()

# Default value.
file_prefix = '2000_10'

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()

for variable in Era5:
  cyclone_tensor_filename = f'{file_prefix}_{variable.name.lower()}_{common.CYCLONE_TENSOR_FILE_POSTFIX}.npy'
  print(f'> loading {cyclone_tensor_filename}')
  cyclone_variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                                                cyclone_tensor_filename)
  cyclone_var_tensor = np.load(file=cyclone_variable_tensor_file_path,\
                               mmap_mode=None, allow_pickle=True)
  no_cyclone_tensor_filename = f'{file_prefix}_{variable.name.lower()}_{common.NO_CYCLONE_TENSOR_FILE_POSTFIX}.npy'
  print(f'> loading {no_cyclone_tensor_filename}')
  no_cyclone_variable_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH,\
                                                   no_cyclone_tensor_filename)
  no_cyclone_var_tensor = np.load(file=no_cyclone_variable_tensor_file_path,\
                               mmap_mode=None, allow_pickle=True)
  print(f'> concatenating the tensors')
  concat_var_tensor = np.concatenate((cyclone_var_tensor, no_cyclone_var_tensor))
  concat_var_tensor_filename = f'{common.MERGED_TENSOR_FILE_PREFIX}_{file_prefix}_{variable.name.lower()}_{common.MERGED_TENSOR_FILE_POSTFIX}.npy'
  concat_var_tensor_file_path = path.join(common.MERGED_TENSOR_PARENT_DIR_PATH,\
                                          concat_var_tensor_filename)
  print(f'> saving {concat_var_tensor_filename} (shape={concat_var_tensor.shape})')
  np.save(file=concat_var_tensor_file_path, arr=concat_var_tensor, allow_pickle=True)

cyclone_filename = f'{file_prefix}_all_{common.CYCLONE_TENSOR_FILE_POSTFIX}.npy'
print(f'> loading {cyclone_filename} and creating the array of labels')
cyclone_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, cyclone_filename)
cyclone_tensor = np.load(file=cyclone_tensor_file_path,\
                               mmap_mode=None, allow_pickle=True)
nb_row_cyclone = cyclone_tensor.shape[0]
labels_cyclone = np.ndarray(shape=(nb_row_cyclone), dtype=np.float32)
labels_cyclone.fill(1.0)

no_cyclone_filename = f'{file_prefix}_all_{common.NO_CYCLONE_TENSOR_FILE_POSTFIX}.npy'
print(f'> loading {no_cyclone_filename} and creating the array of labels')
no_cyclone_tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, no_cyclone_filename)
no_cyclone_tensor = np.load(file=cyclone_tensor_file_path,\
                               mmap_mode=None, allow_pickle=True)
nb_row_no_cyclone = no_cyclone_tensor.shape[0]
labels_no_cyclone = np.ndarray(shape=(nb_row_no_cyclone), dtype=np.float32)
labels_no_cyclone.fill(0.0)

print('> merging the tensor and labels')
merge_tensor = np.concatenate((cyclone_tensor, no_cyclone_tensor))
merge_labels  = np.concatenate((labels_cyclone, labels_no_cyclone))

print('> shuffling the tensor and the labels')
permutation = np.random.permutation((nb_row_cyclone + nb_row_no_cyclone))
shuffled_tensor = merge_tensor[permutation]
shuffled_labels  = merge_labels[permutation]

print(f'> executing {common.STAT_SCRIPT_NAME}')

script_file_path = path.join(common.SCRIPT_DIR_PATH, common.STAT_SCRIPT_NAME)

cmd_line = [script_file_path, f'{common.MERGED_TENSOR_FILE_PREFIX}_{file_prefix}',\
            common.MERGED_TENSOR_FILE_POSTFIX,\
            common.MERGED_TENSOR_PARENT_DIR_PATH, '0']

with subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE,\
                      bufsize=1, universal_newlines=True) as process:
  for line in process.stdout:
    print(line, end='')
  print(f'> stderr: {process.stderr.readlines()}')

shuffled_tensor_filename = f'{common.SHUFFLED_TENSOR_FILE_PREFIX}_{file_prefix}_{common.SHUFFLED_TENSOR_FILE_POSTFIX}.npy'
print(f'> saving {shuffled_tensor_filename}')
shuffled_tensor_file_path = path.join(common.SHUFFLED_TENSOR_PARENT_DIR_PATH, shuffled_tensor_filename)
np.save(file=shuffled_tensor_file_path, arr=shuffled_tensor, allow_pickle=True)

shuffled_labels_filename  = f'{common.SHUFFLED_TENSOR_FILE_PREFIX}_{file_prefix}_labels.npy'
print(f'> saving {shuffled_labels_filename}')
shuffled_labels_file_path = path.join(common.SHUFFLED_TENSOR_PARENT_DIR_PATH, shuffled_labels_filename)
np.save(file=shuffled_labels_file_path, arr=shuffled_labels, allow_pickle=True)

stop = time.time()
print("> spend %f seconds processing"%((stop-start)))