#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:40:14 2018

@author: Sébastien Gardoll
"""

import numpy as np
import pandas as pd
import os.path as path

import sys

import common

import time
start = time.time()

# Default value.
file_prefix = '2000_10'

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
    file_prefix = sys.argv[1].strip()

cyclone_db_file_path = path.join(common.DATASET_PARENT_DIR_PATH,
                                 f'{file_prefix}_{common.CYCLONE_DB_FILE_POSTFIX}.csv')
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
            if common.is_overlapping(lat, lon, lat2, lon2):
                return True
        return False
    else:
        return False


def compute_no_cyclone(time_tuple, delta):
    (year, month, day, hour, lat, lon) = time_tuple
    past = common.subtract_delta(year, month, day, delta)
    has_cyclone = _has_cyclone(past, hour, lat, lon)
    while has_cyclone:
        hour = hour - 6
        if hour < 0:
            hour = int((common.HOURLY_TIME_SAMPLING / common.FOUR_DAILY_TIME_SAMPLING) * 3)
            past = common.subtract_one_day_from_date(past)
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
        no_cyclone_list.append(compute_no_cyclone(cyclone_values, common.ONE_DAY))
        no_cyclone_list.append(compute_no_cyclone(cyclone_values, common.ONE_WEEK))

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

    filename = f'{file_prefix}_{common.NO_CYCLONE_DB_FILE_POSTFIX}.csv'
    print(f'> saving the {filename} on disk')
    no_cyclone_dataframe_file_path = path.join(common.DATASET_PARENT_DIR_PATH,
                                               filename)
    no_cyclone_dataframe.to_csv(no_cyclone_dataframe_file_path, sep=',',
                                na_rep='', header=True, index=True,
                                index_label='id', encoding='utf8',
                                line_terminator='\n')

    stop = time.time()
    formatted_time = common.display_duration((stop-start))
    print(f'> spend {formatted_time} processing')


if __name__ == '__main__':
    main()
    sys.exit(0)
