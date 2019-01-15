import sys
import os.path as path
import numpy as np
import pandas as pd
import xarray as xr
import dask
import common
from common import Era5
import time

                               ##### SETTINGS #####

# Dask scheduler must be threads and mono node as the set of netcdf file
# is shared between the worker threads.
dask.config.set(scheduler='threads')

                              ##### GLOBAL VARS ####

start = time.time()

_TIME_STEP_CONVERTER = {0:'T00:00:00.000000000',
                        1:'T06:00:00.000000000',
                        2:'T12:00:00.000000000',
                        3:'T18:00:00.000000000'}

_LON_CONVERTER = np.load(path.join(common.DATASET_PARENT_DIR_PATH,
                                   'longitude_convert.npy')).item()

                              ##### FUNCTIONS #####

# year_upper_bound is included.
def load_dataset(variable, year_lower_bound, year_upper_bound):
  file_paths = list()
  for year in range(year_lower_bound, (year_upper_bound+1)):
    for month in range(1, 13):
      file_paths.append(variable.value.compute_file_path(year, month))
  return xr.open_mfdataset(paths=file_paths, parallel=True)

def extract_region(dataset, variable, year, month, day, time_step, lat, lon):
  converted_time_step = _TIME_STEP_CONVERTER[time_step]
  rounded_lat = common.round_nearest(lat, common.LAT_RESOLUTION, common.NUM_DECIMAL_LAT)
  rounded_lon = common.round_nearest(lon, common.LON_RESOLUTION, common.NUM_DECIMAL_LON)
  # Latitudes are stored inverted.
  # Minus LAT_RESOLUTION because the upper bound in slice is included.
  lat_min  = (rounded_lat - common.HALF_LAT_FRAME + common.LAT_RESOLUTION)
  lat_max  = (rounded_lat + common.HALF_LAT_FRAME)
  # Minus LON_RESOLUTION because the upper bound in slice is included.
  lon_min  = _LON_CONVERTER[(rounded_lon - common.HALF_LON_FRAME)]
  lon_max  = _LON_CONVERTER[(rounded_lon + common.HALF_LON_FRAME - common.LON_RESOLUTION)]
  if variable.value.level:
    return dataset.sel(time=f'{year}-{month}-{day}{converted_time_step}',
                       level=variable.value.level,
                       latitude=slice(lat_max, lat_min),
                       longitude=slice(lon_min, lon_max))
  else:
    return dataset.sel(time=f'{year}-{month}-{day}{converted_time_step}',
                       latitude=slice(lat_max, lat_min),
                       longitude=slice(lon_min, lon_max))

def load_db(file_path):
  db_file = open(file_path, 'r')
  dataset = pd.read_csv(db_file, sep=',', header=0, index_col=0, na_values='')
  db_file.close()
  return dataset

def cyclone_db_row_processor(row):
  year      =   int(row[3])
  month     =   int(row[4])
  day       =   int(row[5])
  time_step =   int(row[6])
  lat       = float(row[8])
  lon       = float(row[9])
  return (year, month, day, time_step, lat, lon)

def no_cyclone_db_row_processor(row):
  year      =   int(row[1])
  month     =   int(row[2])
  day       =   int(row[3])
  time_step =   int(row[4])
  lat       = float(row[5])
  lon       = float(row[6])
  return (year, month, day, time_step, lat, lon)

def build_channel(nc_dataset, variable, itertuples, row_processor,
                  file_prefix, file_postfix):
  id_name = 'id'
  dim_names = (id_name, 'x', 'y')
  print('  > building the list of subregions')
  region_list= list()
  for row in itertuples:
    (year, month, day, time_step, lat, lon) = row_processor(row)
    current_region = extract_region(nc_dataset, variable, year, month, day,
                                    time_step, lat, lon)
    # Drop netcdf behavior so as to stack the subregions without Na filling
    # (concat netcdf default behavior is to concatenate the subregions on a
    # wilder region that includes all the subregions).
    current_region_data = xr.DataArray(data=current_region.to_array().data,
                                       dims=dim_names)
    region_list.append(current_region_data)

  print(f'  > computing channel {variable.name.lower()}')
  region_stack = xr.concat(objs=region_list, dim=id_name)
  # This instruction is parallelized by xarray/dask.
  #region_stack.compute() # Avoid memory error when np.array().
  channel = np.array(region_stack)
  print(f'  > saving {variable.name.lower()} channel, shape={channel.shape}')
  channel_file_path = path.join(common.CHANNEL_PARENT_DIR_PATH,
        f'{file_prefix}_{variable.name.lower()}_{file_postfix}.npy')
  np.save(file=channel_file_path, arr=channel, allow_pickle=True)
  return True

def test(variable, year_lower_bound, year_upper_bound, year, month, day, time_step, lat, lon):
  nc_dataset = load_dataset(variable, year_lower_bound, year_upper_bound)
  region = extract_region(nc_dataset, variable, year, month, day, time_step, lat, lon)
  region = region.compute() # Avoid memory error when np.array().
  region_npy = np.array(region.to_array()).reshape((common.Y_RESOLUTION,
                                                    common.X_RESOLUTION))
  from matplotlib import pyplot as plt
  plt.figure()
  plt.imshow(region_npy, cmap='gist_rainbow_r',interpolation="none")
  plt.show()
  nc_dataset.close()

def unit_tests():
  test(Era5.MSL, 2000, 2017, 2011, 8, 21, 0, 15, -59)
  test(Era5.TA200, 2000, 2017, 2011, 8, 25, 3, 26.5, -77.2)

                                 ##### MAIN #####

# Default values.
file_prefix = '2000_10'
variable    = Era5.MSL

if (len(sys.argv) > 1) and (sys.argv[1].strip()) and (sys.argv[2].strip()):
  file_prefix = sys.argv[1].strip()
  variable = Era5[sys.argv[2].strip()]

cyclone_db_filename = f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv'
print(f'> loading {cyclone_db_filename}')
cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH, cyclone_db_filename)
cyclone_ds = load_db(cyclone_db_file_path)

no_cyclone_db_filename = f'{file_prefix}_{common.NO_CYCLONE_DB_FILE_POSTFIX}.csv'
print(f'> loading {no_cyclone_db_filename}')
no_cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH, no_cyclone_db_filename)
no_cyclone_ds = load_db(no_cyclone_db_file_path)

lower_year_cyclone    =    cyclone_ds['year'].min()
upper_year_cyclone    =    cyclone_ds['year'].max()
lower_year_no_cyclone = no_cyclone_ds['year'].min()
upper_year_no_cyclone = no_cyclone_ds['year'].max()

year_lower_bound = lower_year_cyclone if lower_year_cyclone < lower_year_no_cyclone \
                   else lower_year_no_cyclone

year_upper_bound = upper_year_cyclone if upper_year_cyclone > upper_year_no_cyclone \
                   else upper_year_no_cyclone

print(f'> lazy loading the netcdf files between {year_lower_bound} and {year_upper_bound}')
# This instruction is parallelized by xarray/dask.
nc_dataset = load_dataset(variable, year_lower_bound, year_upper_bound)

print('> processing the cyclone db')
build_channel(nc_dataset, variable,
              cyclone_ds.itertuples(),
              cyclone_db_row_processor,
              file_prefix,
              common.CYCLONE_CHANNEL_FILE_POSTFIX)

print('> processing the no cyclone db')
build_channel(nc_dataset, variable,
              no_cyclone_ds.itertuples(),
              no_cyclone_db_row_processor,
              file_prefix,
              common.NO_CYCLONE_CHANNEL_FILE_POSTFIX)

nc_dataset.close()

stop = time.time()
formatted_time =common.display_duration((stop-start))
print(f'> spend {formatted_time} processing')