#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:51:31 2018

@author: sebastien@gardoll.fr
"""
import os

import os.path as path
from typing import Tuple, Mapping

import nxtensor.utils.hdf5_utils as h5
import nxtensor.utils.time_utils as tu

import sys

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import time

################################### ENVIRONMENT #######################################

print(f'GPU support: {tf.test.is_built_with_gpu_support()}, CUDA support: {tf.test.is_built_with_cuda()}')
print('GPU availability:')
tf.test.is_gpu_available()

##################################### SETTINGS ########################################

# The settings have been optimized (see optimize_conv_net.py).

# In general: Larger batch sizes result in faster progress in training, but don't always converge as fast and takes
# more memory. Smaller batch sizes train slower, but can converge faster and may have some regularization effects.
# Typical values: 32, 64, 128, 256. High value are reserved for GPU computing (parallel computation of the gradient).
BATCH_SIZE = 32

NUMBER_EPOCHS = 20

LOSS_FUNC = keras.losses.BinaryCrossentropy()  # https://keras.io/losses/
METRICS = ['accuracy']

LEARNING_RATE = 0.01
OPTIMIZER = keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# set data_format to 'channels_last'
keras.backend.set_image_data_format('channels_last')


#################################### CONSTANTS ########################################

DATE_FORMAT = '%m_%d_%Y_%H-%M-%S'

# Default values:
DEFAULT_EXTRACTION_SET_NAME = 'all'
DEFAULT_PREFIX = ''
DEFAULT_PARENT_DIR_PATH = '/data/sgardoll/era5_extractions/all_extraction/tensors'

METADATA_TYPES = {'day': np.int8, 'day2d': np.str,
                  'hour': np.int8, 'hour2d': np.str,
                  'month': np.int8, 'month2d': np.str,
                  'year': np.int16, 'lat': np.float64, 'lon': np.float64,
                  'label_num_id': np.float64}


##################################### FUNCTIONS #######################################


def load_data(data_parent_dir_path: str, data_prefix: str, data_extraction_set_name: str) -> Mapping[str, np.ndarray]:
    training_tensor_file_path = \
        path.join(data_parent_dir_path, f'{data_prefix}training_{data_extraction_set_name}_data.h5')
    training_tensor = h5.read_ndarray_from_hdf5(file_path=training_tensor_file_path)
    validation_tensor_file_path = \
        path.join(data_parent_dir_path, f'{data_prefix}validation_{data_extraction_set_name}_data.h5')
    validation_tensor = h5.read_ndarray_from_hdf5(file_path=validation_tensor_file_path)
    test_tensor_file_path = \
        path.join(data_parent_dir_path, f'{data_prefix}test_{data_extraction_set_name}_data.h5')
    test_tensor = h5.read_ndarray_from_hdf5(file_path=test_tensor_file_path)
    training_labels_file_path = path.join(data_parent_dir_path, f'training_{data_extraction_set_name}_metadata.csv')
    training_metadata: pd.DataFrame = pd.read_csv(filepath_or_buffer=training_labels_file_path, dtype=METADATA_TYPES)
    training_labels = training_metadata['label_num_id'].to_numpy()
    validation_labels_file_path = path.join(data_parent_dir_path, f'validation_{data_extraction_set_name}_metadata.csv')
    validation_metadata: pd.DataFrame = pd.read_csv(filepath_or_buffer=validation_labels_file_path,
                                                    dtype=METADATA_TYPES)
    validation_labels = validation_metadata['label_num_id'].to_numpy()
    test_labels_file_path = path.join(data_parent_dir_path, f'test_{data_extraction_set_name}_metadata.csv')
    test_metadata: pd.DataFrame = pd.read_csv(filepath_or_buffer=test_labels_file_path, dtype=METADATA_TYPES)
    test_labels = test_metadata['label_num_id'].to_numpy()
    return {'training_tensor': training_tensor, 'training_labels': training_labels,
            'validation_tensor': validation_tensor, 'validation_labels': validation_labels,
            'test_tensor': test_tensor, 'test_labels': test_labels}


def create_model(tensor_shape: Tuple[float, float, float]) -> keras.Model:
    input_layer = Input(shape=tensor_shape, name='input')
    conv1_layer = Conv2D(filters=8, kernel_size=(5, 5), activation='relu', name='conv1')(input_layer)
    pooling1_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool1')(conv1_layer)
    conv2_layer = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name='conv2')(pooling1_layer)
    pooling2_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool2')(conv2_layer)
    flatten1_layer = Flatten(name='flatten1')(pooling2_layer)
    fully1_layer = Dense(units=50, activation='relu', name='fully1')(flatten1_layer)
    fully2_layer = Dense(units=1, activation='sigmoid', name='fully2')(fully1_layer)
    model = Model(inputs=input_layer, outputs=fully2_layer)
    return model


def main():
    if (len(sys.argv) > 3) and (sys.argv[1].strip()) and (sys.argv[2].strip()) and (sys.argv[3].strip()):
        data_prefix = sys.argv[1].strip()
        data_extraction_set_name = sys.argv[2].strip()
        data_parent_dir_path = sys.argv[3].strip()
        print(f'> settings prefix to {data_prefix}')
        print(f'> setting parent directory to {data_parent_dir_path}')
    else:
        data_prefix = DEFAULT_PREFIX
        data_extraction_set_name = DEFAULT_EXTRACTION_SET_NAME
        data_parent_dir_path = DEFAULT_PARENT_DIR_PATH
    start = time.time()
    data = load_data(data_parent_dir_path, data_prefix, data_extraction_set_name)
    print(f"training tensor shape: {data['training_tensor'].shape}")
    print(f"validation tensor shape: {data['validation_tensor'].shape}")
    print(f"test tensor shape: {data['test_tensor'].shape}")
    print(f"training labels shape: {data['training_labels'].shape}")
    print(f"validation labels shape: {data['validation_labels'].shape}")
    print(f"test labels shape: {data['test_labels'].shape}")

    model = create_model(data['training_tensor'].shape[1:])
    print('> compiling the layers')
    model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER, metrics=METRICS)

    print(model.summary())

    # FITTING MODEL #######
    import datetime
    log_dir_path = path.join('fit_logs', datetime.datetime.now().strftime(DATE_FORMAT))
    os.makedirs(log_dir_path, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir_path, histogram_freq=1)

    model_filename = f'{data_prefix}best_{data_extraction_set_name}_model_{datetime.datetime.now().strftime(DATE_FORMAT)}.h5'
    model_file_path = path.join('model', model_filename)
    os.makedirs(path.dirname(model_file_path), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(filepath=model_file_path, monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min', period=1)

    # %load_ext tensorboard
    print('> fitting the model')
    model.fit(x=data['training_tensor'], y=data['training_labels'],
              validation_data=(data['validation_tensor'], data['validation_labels']),
              epochs=NUMBER_EPOCHS, batch_size=BATCH_SIZE, verbose=2,
              callbacks=[tensorboard_callback, checkpoint_callback])

    # %tensorboard --logdir $log_dir_path

    print('> evaluating the model on test dataset')
    loss_value, metric_value = model.evaluate(x=data['test_tensor'], y=data['test_labels'], verbose=0)
    print(f'  > loss = {loss_value}')
    print(f'  > metric = {metric_value}')

    test_predicted_probs = model.predict(data['test_tensor'], verbose=0)
    auc_model = roc_auc_score(y_true=data['test_labels'], y_score=test_predicted_probs)
    print(f'  > roc = {auc_model}')

    # Convert the probabilities into the class based on the higher probability.
    # Class 0 for no cyclone, 1 for cyclone.
    threshold_probability = 0.5
    test_predicted_class = np.where(test_predicted_probs > threshold_probability, 1, 0)

    print('  > the classification report:')
    print(classification_report(y_true=data['test_labels'], y_pred=test_predicted_class, target_names=['no_cyclones',
                                                                                                       'cyclones']))
    stop = time.time()
    formatted_time = tu.display_duration((stop-start))
    print(f'> spend {formatted_time} processing')

####################################### MAIN ##########################################


if __name__ == '__main__':
    main()
