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

PLOT_SIZE = (20, 2.5)

# num_viz <= 0 : display all the false negatives.
def rand_display_images(x_test, y_pred_class, y_true_class, y_pred_prob,
                        num_viz, conditions, result_type):
  y_pred_df = pd.DataFrame(data=y_pred_class, columns=['y_pred_class'])
  y_true_df = pd.DataFrame(data=y_true_class, columns=['y_true_class'])
  y_df = pd.concat((y_pred_df, y_true_df), axis=1)
  selected_images = y_df[(y_df.y_pred_class == conditions[0]) & \
                         (y_df.y_true_class == conditions[1])]
  print(f'  > number of selected images: {len(selected_images)}')
  if num_viz > 0:
    subset_selected_images = selected_images.take(range(0,num_viz))
  else:
    subset_selected_images = selected_images
  selected_images_indexes = subset_selected_images.index.values
  display_selected_images(x_test, selected_images_indexes, y_pred_prob, result_type)

def rand_display_true_negatives(x_test, y_pred_class, y_true_class, y_pred_prob,
                                 num_viz):
  conditions = (common.NO_CYCLONE_LABEL, common.NO_CYCLONE_LABEL)
  rand_display_images(x_test, y_pred_class, y_true_class,
                      y_pred_prob, num_viz, conditions, 'True Negative')

def rand_display_false_negatives(x_test, y_pred_class, y_true_class, y_pred_prob,
                                 num_viz):
  conditions = (common.NO_CYCLONE_LABEL, common.CYCLONE_LABEL)
  rand_display_images(x_test, y_pred_class, y_true_class,
                      y_pred_prob, num_viz, conditions, 'False Negative')

def rand_display_true_positives(x_test, y_pred_class, y_true_class, y_pred_prob,
                                num_viz):
  conditions = (common.CYCLONE_LABEL, common.CYCLONE_LABEL)
  rand_display_images(x_test, y_pred_class, y_true_class,
                      y_pred_prob, num_viz, conditions, 'True Positive')

def rand_display_false_positives(x_test, y_pred_class, y_true_class, y_pred_prob,
                                num_viz):
  conditions = (common.CYCLONE_LABEL, common.NO_CYCLONE_LABEL)
  rand_display_images(x_test, y_pred_class, y_true_class,
                      y_pred_prob, num_viz, conditions, 'False Positive')

def display_selected_images(x_test, false_negative_indexes, y_pred_prob,
                            result_type):
  images = x_test[false_negative_indexes]
  image_count = 0
  for image in images:
    probabilities = y_pred_prob[false_negative_indexes[image_count]]
    p_cyclone = probabilities[int(common.CYCLONE_LABEL)]
    p_no_cyclone = probabilities[int(common.NO_CYCLONE_LABEL)]
    image_count = image_count + 1
    print(f'\n\n  > image #{image_count}, {result_type} with probabilities: cyclone {p_cyclone} ;\
 no cyclone {p_no_cyclone}')
    plt.rc('text', usetex=False)
    suptitle = f'{result_type}: p(cyclone) = {common.format_float_number(p_cyclone)} ;\
 p(no cyclone) = {common.format_float_number(p_no_cyclone)}'
    display_image(image, suptitle)

def display_image(image, suptitle):
  reshape_image = image.swapaxes(0,2)
  plt.figure(figsize=PLOT_SIZE)
  for variable in Era5:
    index = variable.value.num_id
    channel = reshape_image[index]
    plt.subplot(1, len(Era5), (index+1))
    plt.title(variable.name.lower(), {'fontsize': 14})
    # Remove ticks from the x and y axes
    plt.xticks([])
    plt.yticks([])
    plt.imshow(channel,cmap='gist_rainbow_r',interpolation="none")
  plt.suptitle(suptitle, fontsize=16, va='bottom')
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

rand_display_true_positives(x_test, y_pred_class, y_test_class, y_pred_prob, 10)
rand_display_true_negatives(x_test, y_pred_class, y_test_class, y_pred_prob, 10)
rand_display_false_negatives(x_test, y_pred_class, y_test_class, y_pred_prob, 0)
rand_display_false_positives(x_test, y_pred_class, y_test_class, y_pred_prob, 0)