#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:51:31 2018

@author: Sébastien Gardoll
"""
import os

import os.path as path

import common

import sys

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten

import time
start = time.time()

# ENVIRONMENT #######

print(f'GPU support: {tf.test.is_built_with_gpu_support()}, CUDA support: {tf.test.is_built_with_cuda()}')
print('GPU availability:')
tf.test.is_gpu_available()

# CONSTANTS #######

DATE_FORMAT = '%m_%d_%Y_%H-%M-%S'

# Default values:
DEFAULT_PREFIX = 'all'
DEFAULT_PARENT_DIR_PATH = '/data/sgardoll/extractions/all_extraction/tensors'

# SETTINGS #######

if (len(sys.argv) > 2) and (sys.argv[1].strip()) and (sys.argv[2].strip()):
    data_prefix = sys.argv[1].strip()
    data_parent_dir_path = sys.argv[2].strip()
    print(f'> settings prefix to {data_prefix}')
    print(f'> setting parent directory to {data_parent_dir_path}')
else:
    data_prefix = DEFAULT_PREFIX
    data_parent_dir_path = DEFAULT_PARENT_DIR_PATH

del DEFAULT_PREFIX, DEFAULT_PARENT_DIR_PATH


# In general: Larger batch sizes result in faster progress in training, but don't always converge as fast.
# Smaller batch sizes train slower, but can converge faster. It's definitely problem dependent.
batch_size = 32 # Default for CPU. # TODO: to be optimzed.

number_epochs = 20  # TODO: to be optimzed.

loss = keras.losses.BinaryCrossentropy()  # https://keras.io/losses/
metrics = ['accuracy']

learning_rate = 0.001  # TODO: Learning rate.
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)  # TODO: choose optimizer.

# set data_format to 'channels_last'
keras.backend.set_image_data_format('channels_last')

# LOADING DATA #######

training_tensor_file_path = path.join(data_parent_dir_path, f'training_{data_prefix}_data.h5')
training_tensor = common.read_ndarray_from_hdf5(filepath=training_tensor_file_path)
print(f'training tensor shape: {training_tensor.shape}')

validation_tensor_file_path = path.join(data_parent_dir_path, f'validation_{data_prefix}_data.h5')
validation_tensor = common.read_ndarray_from_hdf5(filepath=validation_tensor_file_path)
print(f'validation tensor shape: {validation_tensor.shape}')

test_tensor_file_path = path.join(data_parent_dir_path, f'test_{data_prefix}_data.h5')
test_tensor = common.read_ndarray_from_hdf5(filepath=test_tensor_file_path)
print(f'test tensor shape: {test_tensor.shape}')


METADATA_TYPES = {'day': np.int8, 'day2d': np.str,
                  'hour': np.int8, 'hour2d': np.str,
                  'month': np.int8, 'month2d': np.str,
                  'year': np.int16, 'lat': np.float64, 'lon': np.float64,
                  'label_num_id': np.float64}

training_labels_file_path = path.join(data_parent_dir_path, f'training_{data_prefix}_metadata.csv')
training_metadata: pd.DataFrame = pd.read_csv(filepath_or_buffer=training_labels_file_path, dtype=METADATA_TYPES)
training_labels = training_metadata['label_num_id'].to_numpy()
print(f'training labels shape: {training_labels.shape}')

validation_labels_file_path = path.join(data_parent_dir_path, f'validation_{data_prefix}_metadata.csv')
validation_metadata: pd.DataFrame = pd.read_csv(filepath_or_buffer=validation_labels_file_path, dtype=METADATA_TYPES)
validation_labels = validation_metadata['label_num_id'].to_numpy()
print(f'validation labels shape: {validation_labels.shape}')

test_labels_file_path = path.join(data_parent_dir_path, f'test_{data_prefix}_metadata.csv')
test_metadata: pd.DataFrame = pd.read_csv(filepath_or_buffer=test_labels_file_path, dtype=METADATA_TYPES)
test_labels = test_metadata['label_num_id'].to_numpy()
print(f'test labels shape: {test_labels.shape}')

# BUILDING CONV NET #######

input_layer = Input(shape=training_tensor.shape[1:], name='input')

conv1_layer = Conv2D(filters=8, kernel_size=(5, 5), activation='relu',
                     name='conv1')(input_layer)
pooling1_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool1')(conv1_layer)

conv2_layer = Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                     name='conv2')(pooling1_layer)
pooling2_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool2')(conv2_layer)

flatten1_layer = Flatten(name='flatten1')(pooling2_layer)

fully1_layer = Dense(units=50, activation='relu', name='fully1')(flatten1_layer)
fully2_layer = Dense(units=1, activation='sigmoid',
                     name='fully2')(fully1_layer)

model = Model(inputs=input_layer, outputs=fully2_layer)

print('> compiling the layers')
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

print(model.summary())

# FITTING MODEL #######
import datetime
log_dir_path = path.join('fit_logs', datetime.datetime.now().strftime(DATE_FORMAT))
os.makedirs(log_dir_path, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_path, histogram_freq=1)

# %load_ext tensorboard
start = time.time()
print('> fitting the model')
model.fit(x=training_tensor, y=training_labels, validation_data=(validation_tensor, validation_labels),
          epochs=number_epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard_callback])

# %tensorboard --logdir $log_dir_path

print('> evaluating the model on test dataset')
loss, metric = model.evaluate(x=test_tensor, y=test_labels, verbose=0)
print(f'  > loss = {loss}')
print(f'  > metric = {metric}')

test_predicted_probs = model.predict(test_tensor, verbose=0)
auc_model = roc_auc_score(y_true=test_labels, y_score=test_predicted_probs)
print(f'  > roc = {auc_model}')


# Convert the probabilities into the class based on the higher probability.
# Class 0 for no cyclone, 1 for cyclone.
threshold_probability = 0.5
test_predicted_class = np.where(test_predicted_probs > threshold_probability, 1, 0)

print('  > the classification report:')
print(classification_report(y_true=test_labels, y_pred=test_predicted_class, target_names=['no_cyclones', 'cyclones']))

model_filename = f'{data_prefix}_model_{datetime.datetime.now().strftime(DATE_FORMAT)}.h5'
model_file_path = path.join('cnn', model_filename)
os.makedirs(path.dirname(model_file_path), exist_ok=True)
print(f'> saving the model ({model_filename})')
model.save(model_file_path)

stop = time.time()
formatted_time = common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')
