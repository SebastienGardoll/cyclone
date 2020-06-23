#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:40:14 2018

@author: sebastien@gardoll.fr
"""

import numpy as np
import pandas as pd
import os.path as path

import sys

import nxtensor.utils.time_utils as tu

from datetime import timedelta
from datetime import datetime
import time

# NetCDF resolution.
LAT_FRAME = 8
LON_FRAME = 8
FOUR_DAILY_TIME_SAMPLING = 4
HOURLY_TIME_SAMPLING   = 24

ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)

CYCLONE_DB_FILE_POSTFIX         = 'cyclone_dataset'
NO_CYCLONE_DB_FILE_POSTFIX      = 'no_cyclone_dataset'

DEFAULT_FILE_PREFIX = '2000_10'
DEFAULT_DATASET_DIR_PATH = '/data/sgardoll/cyclone_data/dataset'

######## FUNCTIONS ########


def subtract_delta(year, month, day, delta):
    result = datetime(year=year, month=month, day=day) - delta
    return result


def subtract_one_day_from_date(date):
    result = date - ONE_DAY
    return result


def is_overlapping(lat1, lon1, lat2, lon2):
    if abs(lat1-lat2) <= LAT_FRAME:
        if abs(lon1 - lon2) <= LON_FRAME:
            return True
        else:
            return False
    else:
        return False


start = time.time()


if (len(sys.argv) > 2) and (sys.argv[1].strip()) and (sys.argv[2].strip()):
    file_prefix = sys.argv[1].strip()
    dataset_dir_path = sys.argv[2].strip()
else:
    file_prefix = DEFAULT_FILE_PREFIX
    dataset_dir_path = DEFAULT_DATASET_DIR_PATH


cyclone_db_file_path = path.join(dataset_dir_path,
                                 f'{file_prefix}_{CYCLONE_DB_FILE_POSTFIX}.csv')
cyclone_db_file = open(cyclone_db_file_path, 'r')
CYCLONE_DATAFRAME = pd.read_csv(cyclone_db_file, sep=',', header=0, index_col=0,
                                na_values='')
cyclone_db_file.close()
del cyclone_db_file
del cyclone_db_file_path

NO_CYCLONE_DF_COLUMNS = {'year': np.int16,
                         'month': np.int8,
                         'day': np.int8,
                         'hour': np.int8,
                         'lat': np.float64,
                         'lon': np.float64}


def _has_cyclone(date, hour, lat, lon):
    records = CYCLONE_DATAFRAME.query(f'year=={date.year} and\
                                      month=={date.month}\
                                      and day=={date.day} and\
                                      hour=={hour}')
    if not records.empty:
        for (index, record) in records.iterrows():
            lat2 = record["lat"]
            lon2 = record["lon"]
            if is_overlapping(lat, lon, lat2, lon2):
                return True
        return False
    else:
        return False


def compute_no_cyclone(time_tuple, delta):
    (year, month, day, hour, lat, lon) = time_tuple
    past = subtract_delta(year, month, day, delta)
    has_cyclone = _has_cyclone(past, hour, lat, lon)
    while has_cyclone:
        hour = hour - 6
        if hour < 0:
            hour = int((HOURLY_TIME_SAMPLING / FOUR_DAILY_TIME_SAMPLING) * 3)
            past = subtract_one_day_from_date(past)
        has_cyclone = _has_cyclone(past, hour, lat, lon)
    return past.year, past.month, past.day, hour, lat, lon


def main():
    no_cyclone_list = []
    print('> computing the no cyclone records')
    current_year = -1
    for (index, row) in CYCLONE_DATAFRAME.iterrows():
        cyclone_year  = row['year']
        cyclone_month = row['month']
        cyclone_day   = row['day']
        cyclone_hour  = row['hour']
        cyclone_lat   = row['lat']
        cyclone_lon   = row['lon']
        if current_year != cyclone_year:
            current_year = cyclone_year
            print(f'  > compute year: {current_year}')
        cyclone_values = (cyclone_year, cyclone_month, cyclone_day,
                          cyclone_hour, cyclone_lat, cyclone_lon)
        no_cyclone_list.append(compute_no_cyclone(cyclone_values, ONE_DAY))
        no_cyclone_list.append(compute_no_cyclone(cyclone_values, ONE_WEEK))

    # Appending rows one by one in the while loop takes far more time then this.
    no_cyclone_dataframe = pd.DataFrame(data=no_cyclone_list,
                                        columns=list(NO_CYCLONE_DF_COLUMNS.keys()))
    # Specify the schema.
    no_cyclone_dataframe = no_cyclone_dataframe.astype(dtype=NO_CYCLONE_DF_COLUMNS)

    # Remove duplicated rows.
    print(f'> number of records before removing the duplicates: {len(no_cyclone_dataframe)}')
    no_cyclone_dataframe = no_cyclone_dataframe.drop_duplicates()
    print(f'> number of records  AFTER removing the duplicates: {len(no_cyclone_dataframe)}')

    # Sort by date (month) (optimize building channels)
    print('> sorting the rows')
    no_cyclone_dataframe.sort_values(by=['year', 'month', 'day', 'hour'],
                                     ascending=True, inplace=True)

    # Rebuild the ids of the dataframe.
    print('> rebuilding the index of the dataframe')
    no_cyclone_dataframe = no_cyclone_dataframe.reset_index(drop=True)

    filename = f'{file_prefix}_{NO_CYCLONE_DB_FILE_POSTFIX}.csv'
    print(f'> saving the {filename} on disk')
    no_cyclone_dataframe_file_path = path.join(dataset_dir_path,
                                               filename)
    no_cyclone_dataframe.to_csv(no_cyclone_dataframe_file_path, sep=',',
                                na_rep='', header=True, index=True,
                                index_label='id', encoding='utf8',
                                line_terminator='\n')

    stop = time.time()
    formatted_time = tu.display_duration((stop-start))
    print(f'> spend {formatted_time} processing')


if __name__ == '__main__':
    main()
    sys.exit(0)
