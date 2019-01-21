#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:32:21 2019

@author: Sébastien Gardoll
"""

import os.path as path

import csv

from netCDF4 import Dataset

def write_dict_to_csv(filepath, dictionary):
  with open (filepath, 'w') as csv_file:
    csv_writter = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    for key, value in dictionary.items():
      csv_writter.writerow([key, value])


parent_dir_path = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2011'
filename = 'ta.201108.aphe5.GLOBAL_025.nc'
file_path = path.join(parent_dir_path, filename)
nc_dataset = Dataset(file_path, 'r')

ta_indexes = dict()

for index in range(0, len(nc_dataset.variables["level"])):
  ta_indexes[nc_dataset.variables["level"][index].data.item(0)] = index

latitude_indexes = dict()

for index in range(0, len(nc_dataset.variables["latitude"])):
  latitude_indexes[nc_dataset.variables["latitude"][index].data.item(0)] = index

longitude_indexes = dict()

for index in range(0, len(nc_dataset.variables["longitude"])):
  _lon = nc_dataset.variables["longitude"][index].data.item(0)
  if _lon > 180.0:
    _lon = _lon -360.0
  longitude_indexes[_lon] = index


# Build -180 to 180 longitude scale to degrees east (0 to 359.75)
longitude_convert = dict()

for index in range(0, len(nc_dataset.variables["longitude"])):
  _lon_de = nc_dataset.variables["longitude"][index].data.item(0)
  if _lon_de > 180.0:
    _lon = _lon_de -360.0
  else:
    _lon = _lon_de
  longitude_convert[_lon] = _lon_de

nc_dataset.close()

parent_dir_path = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2011'
filename = 'v.201108.aphe5.GLOBAL_025.nc'
file_path = path.join(parent_dir_path, filename)
nc_dataset = Dataset(file_path, 'r')

v_indexes = dict()

for index in range(0, len(nc_dataset.variables["level"])):
  v_indexes[nc_dataset.variables["level"][index].data.item(0)] = index


nc_dataset.close()

parent_dir_path = '/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2011'
filename = 'u.201108.aphe5.GLOBAL_025.nc'
file_path = path.join(parent_dir_path, filename)
nc_dataset = Dataset(file_path, 'r')

u_indexes = dict()

for index in range(0, len(nc_dataset.variables["level"])):
  u_indexes[nc_dataset.variables["level"][index].data.item(0)] = index


nc_dataset.close()

dataset_parent_dir_path='/data/sgardoll/cyclone_data.clean/dataset'

latitude_indexes_file_path=path.join(dataset_parent_dir_path, 'latitude_indexes.csv')
write_dict_to_csv(filepath=latitude_indexes_file_path, dictionary=latitude_indexes)

longitude_indexes_file_path=path.join(dataset_parent_dir_path, 'longitude_indexes.csv')
write_dict_to_csv(filepath=longitude_indexes_file_path, dictionary=longitude_indexes)

longitude_convert_file_path=path.join(dataset_parent_dir_path, 'longitude_convert.csv')
write_dict_to_csv(filepath=longitude_convert_file_path, dictionary=longitude_convert)

v_indexes_file_path=path.join(dataset_parent_dir_path, 'v_indexes.csv')
write_dict_to_csv(filepath=v_indexes_file_path, dictionary=v_indexes)

u_indexes_file_path=path.join(dataset_parent_dir_path, 'u_indexes.csv')
write_dict_to_csv(filepath=u_indexes_file_path, dictionary=u_indexes)

ta_indexes_file_path=path.join(dataset_parent_dir_path, 'ta_indexes.csv')
write_dict_to_csv(filepath=ta_indexes_file_path, dictionary=ta_indexes)