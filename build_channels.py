import multiprocessing as mp
from multiprocessing import Pool
import ctypes
import sys
import os.path as path
import logging
import time

import numpy as np

import pandas as pd

from netCDF4 import Dataset

import common
from common import Era5
import extraction_utils as utils

                              ##### GLOBAL VARS ####

start                = time.time()
cyclone_img_array    = None
no_cyclone_img_array = None

# Args
file_prefix          = '2000_10'
variable             = Era5.MSL
nb_proc              = 5

                              ##### FUNCTIONS #####

def load_db(file_path):
  db_file = open(file_path, 'r')
  # Don't reuse the cyclone id as index.
  dataset = pd.read_csv(db_file, sep=',', header=0, na_values='')
  db_file.close()
  return dataset

def load_netcdf(year, month):
  global variable
  file_path = variable.value.compute_file_path(year, month)
  try:
    result = Dataset(file_path, 'r')
  except Exception as e:
    logging.error(f'> cannot open {file_path}: {str(e)}')
  return result

def cyclone_db_row_processor(row):
  index     =   int(row[0])
  year      =   int(row[4])
  month     =   int(row[5])
  day       =   int(row[6])
  time_step =   int(row[7])
  lat       = float(row[9])
  lon       = float(row[10])
  return (index, year, month, day, time_step, lat, lon)

def no_cyclone_db_row_processor(row):
  index     =   int(row[0])
  year      =   int(row[2])
  month     =   int(row[3])
  day       =   int(row[4])
  time_step =   int(row[5])
  lat       = float(row[6])
  lon       = float(row[7])
  return (index, year, month, day, time_step, lat, lon)

def process_dataframe(img_array, nc_dataset, dataframe, row_processor):
  global variable
  if dataframe is not None:
    for row in dataframe.itertuples():
      (index, year, month, day, time_step, lat, lon) = row_processor(row)
      subregion = utils.extract_region(nc_dataset, variable, day, time_step, lat, lon)
      np.copyto(dst=img_array[index], src=subregion, casting='no')

# Don't print any messages => synchronisation with stdout !!!
def process(item):
  global cyclone_img_array
  global no_cyclone_img_array
  (year, month), month_aggregation = item
  cyclone_df = month_aggregation[0]
  no_cyclone_df = month_aggregation[1]
  nc_dataset = load_netcdf(year, month)
  process_dataframe(cyclone_img_array, nc_dataset, cyclone_df,
                    cyclone_db_row_processor)
  process_dataframe(no_cyclone_img_array, nc_dataset, no_cyclone_df,
                    no_cyclone_db_row_processor)
  nc_dataset.close()
  return True

def save_img_array(img_array, file_postfix):
  global file_prefix
  global variable
  print(f'> saving {variable.name.lower()} channel, shape={img_array.shape}')
  channel_file_path = path.join(common.CHANNEL_PARENT_DIR_PATH,
        f'{file_prefix}_{variable.name.lower()}_{file_postfix}.npy')
  np.save(file=channel_file_path, arr=img_array, allow_pickle=False)

                                 ##### MAIN #####

if (len(sys.argv) > 1) and (sys.argv[1].strip()) and (sys.argv[2].strip()) and (sys.argv[3].strip()):
  file_prefix = sys.argv[1].strip()
  variable = Era5[sys.argv[2].strip()]
  nb_proc = int(sys.argv[3].strip())

cyclone_db_filename = f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv'
print(f'> loading {cyclone_db_filename}')
cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH, cyclone_db_filename)
cyclone_ds = load_db(cyclone_db_file_path)
cyclone_ds_size = len(cyclone_ds)

no_cyclone_db_filename = f'{file_prefix}_{common.NO_CYCLONE_DB_FILE_POSTFIX}.csv'
print(f'> loading {no_cyclone_db_filename}')
no_cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH, no_cyclone_db_filename)
no_cyclone_ds = load_db(no_cyclone_db_file_path)
no_cyclone_ds_size = len(no_cyclone_ds)

processing_dict = dict()
for by in cyclone_ds.groupby(['year', 'month']):
  (year, month), group = by
  processing_dict[(year, month)]=[group, None]

for by in no_cyclone_ds.groupby(['year', 'month']):
  (year, month), group = by
  if not (year, month) in processing_dict:
    processing_dict[(year, month)] = [None, None]
  list = processing_dict[(year, month)]
  list[1] = group

cyclone_img_array = np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(ctypes.c_float, common.Y_RESOLUTION), common.X_RESOLUTION), cyclone_ds_size))
no_cyclone_img_array = np.ctypeslib.as_array(mp.RawArray(ctypes.ARRAY(ctypes.ARRAY(ctypes.c_float, common.Y_RESOLUTION), common.X_RESOLUTION), no_cyclone_ds_size))

print(f'> processing variable {variable.name} on dataset {file_prefix} with {nb_proc} workers')
with Pool(processes = nb_proc) as pool:
  pool.map(func=process, iterable=processing_dict.items(), chunksize=1)

save_img_array(cyclone_img_array, common.CYCLONE_CHANNEL_FILE_POSTFIX)
save_img_array(no_cyclone_img_array, common.NO_CYCLONE_CHANNEL_FILE_POSTFIX)

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')