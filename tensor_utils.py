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

import keras

# num_viz <= 0 : display all the false negatives.
def rand_display_false_negatives(x_test, y_pred_class, y_true_class, y_pred_prob,
                                 num_viz):
  y_pred_df = pd.DataFrame(data=y_pred_class, columns=['y_pred_class'])
  y_true_df = pd.DataFrame(data=y_true_class, columns=['y_true_class'])
  y_df = pd.concat((y_pred_df, y_true_df), axis=1)
  false_negatives = y_df[(y_df.y_pred_class == common.NO_CYCLONE_LABEL) & \
                         (y_df.y_true_class == common.CYCLONE_LABEL)]
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
    probabilities = y_pred_prob[false_negative_indexes[image_count]]
    image_count = image_count + 1
    print(f'\n\n> image #{image_count}, probabilities: cyclone\
 {probabilities[int(common.CYCLONE_LABEL)]} ;\
 no cyclone {probabilities[int(common.NO_CYCLONE_LABEL)]}')
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

def load_results(file_prefix):
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
  return (x_test, y_test_class, y_pred_class, y_pred_prob)

def load_model(file_prefix):
  model_filename  = f'{file_prefix}_model.h5'
  model_file_path = path.join(common.CNN_PARENT_DIR_PATH, model_filename)
  print(f'> loading model {model_filename}')
  return keras.models.load_model(model_file_path)

                      ####### MAIN #######

file_prefix = '2kb'

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()
  print(f'> setting file prefix to {file_prefix}')


(x_test, y_test_class, y_pred_class, y_pred_prob) = load_results(file_prefix)

rand_display_false_negatives(x_test, y_pred_class, y_test_class, y_pred_prob, 1)
