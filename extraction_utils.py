#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:33:54 2018

@author: Sébastien Gardoll
"""

                        ######## IMPORTS ########
import common

from common import Era5

from netCDF4 import Dataset

import os.path as path

from matplotlib import pyplot as plt

import numpy as np

import logging

# Internal static variables

_LATITUDE_INDEXES  = common.read_dict_from_csv(
                      path.join(common.DATASET_PARENT_DIR_PATH,
                               'latitude_indexes.csv'), float, int)
_LONGITUDE_INDEXES = common.read_dict_from_csv(
                      path.join(common.DATASET_PARENT_DIR_PATH,
                               'longitude_indexes.csv'), float, int)

                       ######## FUNCTIONS ########


def _compute_time_index(num_day, time_step = 0):
  if time_step >= common.TIME_SAMPLING :
    days_to_add = int(time_step / common.TIME_SAMPLING)
    time_step = time_step % common.TIME_SAMPLING
    num_day = num_day + days_to_add
  return common.TIME_SAMPLING*(num_day-1) + time_step


def extract_region(nc_dataset, variable, day, time_step, lat, lon):
  rounded_lat = common.round_nearest(lat, common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
  rounded_lon = common.round_nearest(lon, common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
  time_min_index =  _compute_time_index(day, time_step)
  time_max_index =  time_min_index + 1
  # latitudes are stored inverted.
  lat_min_index  = _LATITUDE_INDEXES[(rounded_lat + common.HALF_LAT_FRAME)]
  lat_max_index  = _LATITUDE_INDEXES[(rounded_lat - common.HALF_LAT_FRAME)]
  lon_min_index  = _LONGITUDE_INDEXES[(rounded_lon - common.HALF_LON_FRAME)]
  lon_max_index  = _LONGITUDE_INDEXES[(rounded_lon + common.HALF_LON_FRAME)]
  #DEBUG
  #print(f'lat_min_index: {lat_min_index} ; lat_max_index: {lat_max_index} ; lon_min_index: {lon_min_index} ; lon_max_index: {lon_max_index}')
  if variable.value.level is None:
    result = nc_dataset[variable.value.str_id][time_min_index:time_max_index,
                       lat_min_index:lat_max_index, lon_min_index:lon_max_index][0]
  else:
    level_index = variable.value.index_mapping[variable.value.level]
    result = nc_dataset[variable.value.str_id][time_min_index:time_max_index,
                        level_index, lat_min_index:lat_max_index,
                        lon_min_index:lon_max_index][0]
  return result


def _compute_netcdf_file_path(parent_dir_path, variable, year, month):
  if variable.value.level is None:
    delta = 'ashe5'
    rep = 'AN_SF'
  else:
    delta = 'aphe5'
    rep = 'AN_PL'
  result = f'{parent_dir_path}/{rep}/{year}/{variable.value.str_id}.{year}{month:02d}.{delta}.GLOBAL_025.nc'
  return result

def open_netcdf(parent_dir_path, variable, year, month, lazy_loading=True):
  file_path = _compute_netcdf_file_path(parent_dir_path, variable, year, month)
  try:
    if lazy_loading:
      result = Dataset(file_path, 'r')
    else:
      netcdf_dataset = Dataset(file_path, 'r')
      if variable.value.level is None:
        netcdf_data = netcdf_dataset[variable.value.str_id]
      else:
        level_index = variable.value.index_mapping[variable.value.level]
        netcdf_data = netcdf_dataset[variable.value.str_id][:, level_index, :, :]
      result = np.array(netcdf_data)
      netcdf_dataset.close()
  except Exception as e:
    logging.error(f'> cannot open {file_path}: {str(e)}')
  return result

def build_dataset_dict(year, month):
  parent_dir_path = common.NETCDF_PARENT_DIR_PATH
  result = {Era5.MSL  : open_netcdf(parent_dir_path, Era5.MSL, year, month),
            Era5.U10  : open_netcdf(parent_dir_path, Era5.U10, year, month),
            Era5.V10  : open_netcdf(parent_dir_path, Era5.V10, year, month),
            Era5.TCWV : open_netcdf(parent_dir_path, Era5.TCWV, year, month),
            Era5.TA200: open_netcdf(parent_dir_path, Era5.TA200, year, month),
            Era5.TA500: open_netcdf(parent_dir_path, Era5.TA500, year, month),
            Era5.U850 : open_netcdf(parent_dir_path, Era5.U850, year, month),
            Era5.V850 : open_netcdf(parent_dir_path, Era5.V850, year, month)}
  return result

def display_region(netcdf_dataset, variable, day, time_step, lat, lon):
  region = extract_region(netcdf_dataset, variable, day, time_step, lat, lon)
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

def close_dataset_dict(dataset_dict):
  for dataset in dataset_dict.values():
    dataset.close()

                           ######## TESTS ########

"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_SF/2011/msl.201108.ashe5.GLOBAL_025.nc"
set region/x=63W:55W/y=11N:19N/t=1855152/k=1
shade msl
"""


def test1():
  parent_dir_path = common.NETCDF_PARENT_DIR_PATH
  variable = Era5.MSL
  year = 2011
  month = 8
  day = 21
  lat = 15
  lon = -59
  time_step = 0
  nc_dataset = open_netcdf(parent_dir_path, variable, year, month)
  region = extract_region(nc_dataset, variable, day, time_step, lat, lon)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()

"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2011/ta.201108.aphe5.GLOBAL_025.nc"
set region/x=81.25W:73.25W/y=22.5N:30.5N/t=1855266/k=15
shade ta
"""


def test2():
  parent_dir_path = common.NETCDF_PARENT_DIR_PATH
  variable = Era5.TA200
  year = 2011
  month = 8
  day = 25
  lat = 26.5
  lon = -77.2
  time_step = 3
  nc_dataset = open_netcdf(parent_dir_path, variable, year, month)
  region = extract_region(nc_dataset, variable, day, time_step, lat, lon)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()
