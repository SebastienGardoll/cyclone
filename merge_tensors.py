#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:58:15 2018

@author: Sébastien Gardoll
"""
import os.path as path
import os

import sys

import common
from common import Era5

import numpy as np

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
  concat_var_tensor_filename = f'merged_{file_prefix}_{variable.name.lower()}_{common.MERGED_TENSOR_FILE_POSTFIX}.npy'
  concat_var_tensor_file_path = path.join(common.MERGED_TENSOR_PARENT_DIR_PATH,\
                                          concat_var_tensor_filename)
  print(f'> saving {concat_var_tensor_filename} (shape={concat_var_tensor.shape})')
  np.save(file=concat_var_tensor_file_path, arr=concat_var_tensor, allow_pickle=True)

