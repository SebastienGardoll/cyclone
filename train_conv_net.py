#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:51:31 2018

@author: Sébastien Gardoll
"""
import os

import os.path as path

import psutil

import common

import sys

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model

import time
start = time.time()

                           ####### SETTINGS #######

file_prefix = '2kb'
num_threads = 4

if (len(sys.argv) > 2) and (sys.argv[1].strip()) and (sys.argv[2].strip()):
  file_prefix = sys.argv[1].strip()
  num_threads = int(sys.argv[2].strip())
  print(f'> setting file prefix to {file_prefix}')
  print(f'> setting number of core to {num_threads}')

num_classes = common.NUM_CLASSES
max_mem     = -1

# TODO optimize settings.
batch_size  = 5
epochs      = 75
loss        = keras.losses.binary_crossentropy # https://keras.io/losses/
metrics     = ['accuracy']
optimizer   = keras.optimizers.SGD() # https://keras.io/optimizers/
#optimizer   = keras.optimizers.Adadelta()
test_ratio  = 0.3

if num_threads > 1:
  num_threads = num_threads - 1

config = K.tf.ConfigProto()#device_count={"CPU":num_threads})

# maximum number of threads per core ?
config.intra_op_parallelism_threads = 1

# maximum number of core ?
config.inter_op_parallelism_threads = num_threads

# maximum is not the actual number of threads/cores used !

K.set_session(K.tf.Session(config=config))

# set data_format to 'channels_last'
keras.backend.set_image_data_format('channels_last')

                         ####### LOADING DATA #######

tensor_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_TENSOR_FILE_POSTFIX}.h5'
tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, tensor_filename)

print(f'> loading tensor {tensor_filename}')
tensor = common.read_ndarray_from_hdf5(filepath=tensor_file_path)

labels_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
labels_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, labels_filename)

print(f'> loading labels {labels_filename}')
labels = common.read_ndarray_from_hdf5(filepath=labels_file_path)

process = psutil.Process(os.getpid())
current_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
if current_mem > max_mem:
  max_mem = current_mem

print('> making test and training datasets')
# Split arrays or matrices into random train and test subsets.
x_train, x_test, y_train_class, y_test_class = train_test_split(tensor,
                                                  labels, test_size=test_ratio)
del tensor, labels

y_train_cat = keras.utils.to_categorical(y_train_class, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test_class, num_classes)

print('> saving the datasets')

file_name = f'train_{file_prefix}_{common.SHUFFLED_TENSOR_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
common.write_ndarray_to_hdf5(file_path, x_train)

file_name = f'train_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
common.write_ndarray_to_hdf5(file_path, y_train_class)

file_name = f'test_{file_prefix}_{common.SHUFFLED_TENSOR_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
common.write_ndarray_to_hdf5(file_path, x_test)

file_name = f'test_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
common.write_ndarray_to_hdf5(file_path, y_test_class)

del file_name
del file_path

process = psutil.Process(os.getpid())
current_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
if current_mem > max_mem:
  max_mem = current_mem

                    ####### BUILDING CONV NET #######

input_layer    = Input(shape=x_train.shape[1:], name='input')

conv1_layer    = Conv2D(filters=8, kernel_size=(5, 5), activation='relu',
                        name='conv1')(input_layer)
pooling1_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool1')(conv1_layer)

conv2_layer    = Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                        name='conv2')(pooling1_layer)
pooling2_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool2')(conv2_layer)

flatten1_layer = Flatten(name='flatten1')(pooling2_layer)

fully1_layer   = Dense(units=50, activation='relu', name='fully1')(flatten1_layer)
fully2_layer   = Dense(units=num_classes, activation='sigmoid',
                       name='fully2')(fully1_layer)

model = Model(inputs=input_layer, outputs=fully2_layer)

print('> compiling the layers')
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

print(model.summary())

cnn_architecture_filename  = 'cnn_layers.png'
cnn_architecture_file_path = path.join(common.CNN_PARENT_DIR_PATH,
                                       cnn_architecture_filename)
print(f'> saving the architecture plot ({cnn_architecture_filename})')
plot_model(model, to_file=cnn_architecture_file_path, show_shapes=True,
           show_layer_names=True)

                      ####### FITTING MODEL #######

# examples per second: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_train.py

print('> fitting the model')
model.fit(x=x_train, y=y_train_cat, epochs=epochs, batch_size=batch_size, verbose=2)

print('> evaluating the model (keras method)')
loss, metric = model.evaluate(x=x_test, y=y_test_cat, verbose=1)
print(f'  > loss = {loss}')
print(f'  > metric = {metric}')

print('> computing AUC')

# Return the probability associated to the class:
# the probability to be a no cyclone for index 0,
# the probability to be a cyclone for index 1.
y_pred_raw = model.predict(x_test, verbose=1)

# Convert the probabilities into the class based on the higher probability.
# Class 0 for no cyclone, 1 for cyclone.
y_pred_class = np.argmax(y_pred_raw, axis=1)

# Keep only the probabilities of cyclones (see roc_auc_score).
y_pred_prob = np.delete(y_pred_raw, obj=0, axis=1).squeeze()

# From scikit learn:
# " For binary y_true, y_score is supposed to be the score of the class with greater label."
auc_model = roc_auc_score(y_true=y_test_class, y_score=y_pred_prob)
print(f'  > {auc_model}')

print('  > displaying the classification report')
print(classification_report(y_true=y_test_class, y_pred=y_pred_class, target_names=('no_cyclones', 'cyclones')))

model_filename  = f'{file_prefix}_model.h5'
model_file_path = path.join(common.CNN_PARENT_DIR_PATH, model_filename)
print(f'> saving the model ({model_filename})')
model.save(model_file_path)

print(f'> saving the predictions')
file_name = f'prediction_class_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
common.write_ndarray_to_hdf5(file_path, y_pred_class)
file_name = f'prediction_prob_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.h5'
file_path = path.join(common.TENSOR_PARENT_DIR_PATH, file_name)
common.write_ndarray_to_hdf5(file_path, y_pred_raw)

del file_name
del file_path

process = psutil.Process(os.getpid())
current_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
if current_mem > max_mem:
  max_mem = current_mem
print(f'> maximum memory footprint: {max_mem:.2f} MiB')

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')
