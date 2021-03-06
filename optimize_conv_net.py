#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  21 10:10:23 2020

@author: sebastien@gardoll.fr
"""

import sys

import train_conv_net as train_conv

import kerastuner as kt

import tensorflow.keras as keras

import datetime


#################################### CLASSES ########################################

class RandomSearchWithBatchSize(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32, default=32)
        super(RandomSearchWithBatchSize, self).run_trial(trial, *args, **kwargs)


#################################### FUNCTIONS ########################################

def build_model(hp: kt.HyperParameters) -> keras.Model:
    model = train_conv.create_model(SHAPE)
    optimizer_class = OPTIMIZERS[hp.Choice(name='optimizer', values=('adam', 'sgd'), default='adam')]
    optimizer = optimizer_class(learning_rate=hp.Choice(name='learning_rate', values=(0.0001, 0.001, 0.01),
                                                        default=0.0001))
    model.compile(loss=train_conv.LOSS_FUNC, optimizer=optimizer, metrics=train_conv.METRICS)
    return model


#################################### CONSTANTS ########################################

OPTIMIZERS = {'adam': keras.optimizers.Adam, 'sgd': keras.optimizers.SGD}
MAX_EPOCH_SEARCH = 20
EXECUTION_PER_TRIAL = 3
SEED = 1


#################################### MAIN ########################################


if (len(sys.argv) > 2) and (sys.argv[1].strip()) and (sys.argv[2].strip()):
    data_prefix = sys.argv[1].strip()
    data_parent_dir_path = sys.argv[2].strip()
    print(f'> settings prefix to {data_prefix}')
    print(f'> setting parent directory to {data_parent_dir_path}')
else:
    data_prefix = train_conv.DEFAULT_PREFIX
    data_parent_dir_path = train_conv.DEFAULT_PARENT_DIR_PATH

data = train_conv.load_data(data_parent_dir_path, data_prefix)

SHAPE = data['training_tensor'].shape[1:]

project_name = f'cyclone_{datetime.datetime.now().strftime(train_conv.DATE_FORMAT)}'
# Max trials is exactly 48 and giving 48 is just making random search a grid optimization search.
tuner = RandomSearchWithBatchSize(build_model, max_trials=48, objective='val_loss', seed=SEED,
                                  executions_per_trial=2, directory='grid', project_name=project_name,
                                  overwrite=True)

tuner.search_space_summary()

tuner.search(x=data['training_tensor'], y=data['training_labels'],
             validation_data=(data['validation_tensor'], data['validation_labels']),
             epochs=20, verbose=2)

tuner.results_summary()
# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model.
loss, accuracy = best_model.evaluate(x=data['test_tensor'], y=data['test_labels'], verbose=0)
print(f'> the best model, loss: {loss} ; accuracy: {accuracy}')

print(f'> the best hyperparameters: {tuner.get_best_hyperparameters()[0].values}')
