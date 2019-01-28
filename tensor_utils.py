#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:55:01 2019

@author: Sébastien Gardoll
"""

import sys

import os.path as path

import pandas as pd

import common
from common import Era5

from matplotlib import pyplot as plt

# num_viz <= 0 : display all the false negatives.
def rand_display_false_negatives(x_test, y_pred_class, y_true_class, y_pred_prob,
                                 num_viz):
  y_pred_df = pd.DataFrame(data=y_pred_class, columns=['y_pred_class'])
  y_true_df = pd.DataFrame(data=y_true_class, columns=['y_true_class'])
  y_df = pd.concat((y_pred_df, y_true_df), axis=1)
  false_negatives = y_df[(y_df.y_pred_class == 0) & (y_df.y_true_class == 1)]
  print(f'> number of false negatives: {len(false_negatives)}')
  if num_viz > 0:
    subset_false_negatives = false_negatives.take(range(0,num_viz))
  else:
    subset_false_negatives = false_negatives
  false_negative_indexes = subset_false_negatives.index.values
  display_false_negatives(x_test, false_negative_indexes, y_pred_prob)

def display_false_negatives(x_test, false_negative_indexes, y_pred_prob):
  images = x_test[false_negative_indexes]
  image_count = 0
  for image in images:
    probability = y_pred_prob[false_negative_indexes[image_count]]
    image_count = image_count + 1
    print(f'\n\n> image #{image_count} with prob {probability}')
    display_image(image)

def display_image(image):
  reshape_image = image.swapaxes(0,2)
  for variable in Era5:
    index = variable.value.num_id
    print(f'\n  > display {variable.name.lower()}')
    channel = reshape_image[index]
    display_channel(channel)

def display_channel(channel):
  plt.figure()
  plt.imshow(channel,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

                      ####### MAIN #######

file_prefix = '2kb'

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()
  print(f'> setting file prefix to {file_prefix}')

file_name = f'test_{file_prefix}_{common.SHUFFLED_TENSOR_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
print(f'> loading {file_name}')
x_test = common.read_ndarray_from_hdf5(file_path)

file_name = f'test_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
print(f'> loading {file_name}')
y_test_class = common.read_ndarray_from_hdf5(file_path)

file_name = f'prediction_class_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
print(f'> loading {file_name}')
y_pred_class = common.read_ndarray_from_hdf5(file_path)

file_name = f'prediction_prob_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
print(f'> loading {file_name}')
y_pred_prob = common.read_ndarray_from_hdf5(file_path)

rand_display_false_negatives(x_test, y_pred_class, y_test_class, y_pred_prob, 1)
