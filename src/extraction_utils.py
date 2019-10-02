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

def extract_region(nc_dataset, variable, day, hour, lat, lon):
  rounded_lat = common.round_nearest(lat, common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
  rounded_lon = common.round_nearest(lon, common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
  time_min_index =  variable.value.compute_time_index(day, hour)
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


def open_netcdf(variable, year, month, lazy_loading=True):
  file_path = variable.value.compute_file_path(year, month)
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
  result = {Era5.MSL  : open_netcdf(Era5.MSL, year, month),
            Era5.U10  : open_netcdf(Era5.U10, year, month),
            Era5.V10  : open_netcdf(Era5.V10, year, month),
            Era5.TCWV : open_netcdf(Era5.TCWV, year, month),
            Era5.TA200: open_netcdf(Era5.TA200, year, month),
            Era5.TA500: open_netcdf(Era5.TA500, year, month),
            Era5.U850 : open_netcdf(Era5.U850, year, month),
            Era5.V850 : open_netcdf(Era5.V850, year, month)}
  return result

def display_region(netcdf_dataset, variable, day, hour, lat, lon):
  region = extract_region(netcdf_dataset, variable, day, hour, lat, lon)
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()

def close_dataset_dict(dataset_dict):
  for dataset in dataset_dict.values():
    dataset.close()

                           ######## TESTS ########

# This test corresponds to the location of the cyclone AL132000,2000,10,1,0,HU,39.7,-47.9
# Ferret and this API use lat/lon rounded to 39.75 and -48.
# Ferret is inclusive whereas this API is not, that why the coordinates in ferret
# are minus 0.25 (instead of 52W:44W and 35.75N:43.75).
# @see implementation details of the function extract_region (slicing is not
# inclusive.
"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2000/msl.200010.as1e5.GLOBAL_025.nc"
set region/x=52W:44.25W/y=36N:43.75N
shade msl[l=1]
list/precision=8/width=1000
"""
def test3():
  variable = Era5.MSL
  year = 2000
  month = 10
  day = 1
  lat = 39.7
  lon = -47.9
  hour = 0
  nc_dataset = open_netcdf(variable, year, month)
  region = extract_region(nc_dataset, variable, day, hour, lat, lon)
  np.set_printoptions(threshold=np.inf)
  print(region)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()


# Same comments as test3.
"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2011/ta.201108.aphe5.GLOBAL_025.nc"
set region/x=81.25W:73.5W/y=22.75N:30.5N/t=1855266/k=15
shade ta
list/precision=8/width=1000
"""
def test2():
  variable = Era5.TA200
  year = 2011
  month = 8
  day = 25
  lat = 26.5
  lon = -77.2
  hour = 18
  nc_dataset = open_netcdf(variable, year, month)
  region = extract_region(nc_dataset, variable, day, hour, lat, lon)
  np.set_printoptions(threshold=np.inf)
  print(region)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()


# Same comments as test3.
"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2011/msl.201108.as1e5.GLOBAL_025.nc"
set region/x=63W:55.25W/y=11.25N:19N/t=1855152/k=1
shade msl
list/precision=8/width=1000
"""
def test1():
  variable = Era5.MSL
  year = 2011
  month = 8
  day = 21
  lat = 15
  lon = -59
  hour = 0
  nc_dataset = open_netcdf(variable, year, month)
  region = extract_region(nc_dataset, variable, day, hour, lat, lon)
  np.set_printoptions(threshold=np.inf)
  print(region)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()

# Same comments as test3.
"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2011/msl.201108.as1e5.GLOBAL_025.nc"
set region/x=63W:55.25W/y=11.25N:19N
shade msl[l=231]
list/precision=8/width=1000
"""
def test0():
  variable = Era5.MSL
  year = 2011
  month = 8
  day = 10
  lat = 15
  lon = -59
  hour = 14
  nc_dataset = open_netcdf(variable, year, month)
  region = extract_region(nc_dataset, variable, day, hour, lat, lon)
  np.set_printoptions(threshold=np.inf)
  print(region)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()

# Same comments as test3.
"""
ferret
use "/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2000/ta.200009.aphe5.GLOBAL_025.nc"
set region/x=81.25W:73.5W/y=22.75N:30.5N/k=15
shade ta[l=108]
list/precision=8/width=1000
"""
def test4():
  variable = Era5.TA200
  year = 2000
  month = 9
  day = 27
  lat = 26.5
  lon = -77.2
  hour = 18
  nc_dataset = open_netcdf(variable, year, month)
  region = extract_region(nc_dataset, variable, day, hour, lat, lon)
  np.set_printoptions(threshold=np.inf)
  print(region)
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region,cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()