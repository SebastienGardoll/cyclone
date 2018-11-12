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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model

import time
start = time.time()

                           ####### SETTINGS #######

file_prefix = '2k'
num_core    = 0
num_classes = common.NUM_CLASSES
max_mem     = -1

# TODO optimize settings.
batch_size  = 5
epochs      = 100
loss        = keras.losses.binary_crossentropy # https://keras.io/losses/
metrics     = ['accuracy']
optimizer   = keras.optimizers.SGD() # https://keras.io/optimizers/
#optimizer   = keras.optimizers.Adadelta()
test_ratio  = 0.3

config = K.tf.ConfigProto()

config.intra_op_parallelism_threads = num_core
config.inter_op_parallelism_threads = num_core

K.set_session(K.tf.Session(config=config))

# set data_format to 'channels_last'
keras.backend.set_image_data_format('channels_last')

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()

                         ####### LOADING DATA #######

tensor_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_TENSOR_FILE_POSTFIX}.npy'
tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, tensor_filename)

print(f'> loading tensor {tensor_filename}')
tensor = np.load(file=tensor_file_path, mmap_mode=None, allow_pickle=True)

labels_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_\
{common.SHUFFLED_LABELS_FILE_POSTFIX}.npy'
labels_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, labels_filename)

print(f'> loading labels {labels_filename}')
labels = np.load(file=labels_file_path, mmap_mode=None, allow_pickle=True)

process = psutil.Process(os.getpid())
current_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
if current_mem > max_mem:
  max_mem = current_mem

print('> making test and training datasets')
x_train, x_test, y_train_not_cat, y_test_not_cat = train_test_split(tensor,
                                                  labels, test_size=test_ratio)
del tensor, labels

y_train = keras.utils.to_categorical(y_train_not_cat, num_classes)
y_test  = keras.utils.to_categorical(y_test_not_cat, num_classes)
del y_train_not_cat


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

print('> fitting the model')
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)

print('> evaluating the model (keras method)')
loss, metric = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f'  > loss = {loss}')
print(f'  > metric = {metric}')

print('> computing AUC')

y_pred = model.predict(x_test, verbose=1)
# Return the class (0 for no cyclone, 1 for cyclone).
y_pred = np.argmax(y_pred, axis=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_not_cat, y_pred)
auc_model = auc(fpr_keras, tpr_keras)
print(f'  > {auc_model}')

model_filename  = 'model.h5'
model_file_path = path.join(common.CNN_PARENT_DIR_PATH, model_filename)
print(f'> saving the model ({model_filename})')
model.save(model_file_path)

process = psutil.Process(os.getpid())
current_mem = process.memory_info().rss/common.MEGA_BYTES_FACTOR
if current_mem > max_mem:
  max_mem = current_mem
print(f'> maximum memory footprint: {max_mem:.2f} MiB')

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')