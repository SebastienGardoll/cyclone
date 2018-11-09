#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:51:31 2018

@author: Sébastien Gardoll
"""

import os.path as path

import common

import sys

import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D
from keras.utils import plot_model

                           ####### SETTINGS #######

file_prefix = '2000_10'
num_core    = 0
num_classes = common.NUM_CLASSES

# TODO optimize settings.
batch_size  = 32
epochs      = 10
loss        = keras.losses.binary_crossentropy # https://keras.io/losses/
metrics     = ['accuracy', 'binary_accuracy']
optimizer   = keras.optimizers.SGD() # https://keras.io/optimizers/

config = K.tf.ConfigProto()

config.intra_op_parallelism_threads = num_core
config.inter_op_parallelism_threads = num_core

K.set_session(K.tf.Session(config=config))

# set data_format to 'channels_last'
keras.backend.set_image_data_format('channels_last')

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  file_prefix = sys.argv[1].strip()


                         ####### LOADING DATA #######

tensor_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_{common.SHUFFLED_TENSOR_FILE_POSTFIX}.npy'
tensor_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, tensor_filename)

print(f'> loading tensor {tensor_filename}')
tensor = np.load(file=tensor_file_path, mmap_mode=None, allow_pickle=True)

labels_filename = f'{common.SHUFFLED_FILE_PREFIX}_{file_prefix}_{common.SHUFFLED_LABELS_FILE_POSTFIX}.npy'
labels_file_path = path.join(common.TENSOR_PARENT_DIR_PATH, labels_filename)

print(f'> loading labels {labels_filename}')
labels = np.load(file=labels_file_path, mmap_mode=None, allow_pickle=True)

                    ####### BUILDING CONV NET #######

input_layer    = Input(shape=tensor.shape[1:], name='input')

conv1_layer    = Conv2D(filters=8, kernel_size=(5, 5), activation='relu',
                        name='conv1')(input_layer)
pooling1_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool1')(conv1_layer)

conv2_layer    = Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                        name='conv2')(pooling1_layer)
pooling2_layer = MaxPooling2D(pool_size=(2, 2), name='max_pool2')(conv2_layer)

fully1_layer   = Dense(units=50, activation='relu', name='fully1')(pooling2_layer)
fully2_layer   = Dense(units=num_classes, activation='sigmoid',
                       name='fully2')(fully1_layer)

model = Model(inputs=input_layer, outputs=fully2_layer)

print('> compiling the layers')
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

cnn_architecture_filename  = 'cnn_layers.png'
cnn_architecture_file_path = path.join(common.CNN_PARENT_DIR_PATH,
                                       cnn_architecture_filename)
print(f'> saving the architecture plot ({cnn_architecture_filename})')
plot_model(model, to_file=cnn_architecture_file_path, show_shapes=True,
           show_layer_names=True)

                      ####### FITTING MODEL #######

model.fit(x=tensor, y=labels, epochs=epochs, batch_size=batch_size)

