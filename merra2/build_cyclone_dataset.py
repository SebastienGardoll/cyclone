#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:36:59 2018

@author: sebastien@gardoll.fr
"""

import re
import pandas as pd
import logging
import os.path as path
import sys

import time
start = time.time()

CYCLONE_HEADER_PATTERN = re.compile('^([A-Z0-9]+), +[-\w]+, +(\d+),$')
CYCLONE_DESC_PATTERN = re.compile('^(\d{4})(\d{2})(\d{2}), (\d{4}), ([ A-Z]), ([A-Z]{2}), +(\d+\.\d+)([NS]), +(\d+\.\d+)([WE]), +(-?[\d ]+), +(-?[\d ]+).+$')

HOUR_MAPPING = {'0000': 0, '0600': 6, '1200': 12, '1800': 18}

CYCLONE_DF_COLUMNS = ['cyclone_id', 'hurdat2_id', 'year', 'month', 'day',
                      'hour', 'status', 'lat', 'lon', 'max_sustained_wind', 'min_pressure']

MERRA2_MIN_YEAR = 1980

MERRA2_LAT_COORDINATES_RESOLUTION = 0.5
MERRA2_LON_COORDINATES_RESOLUTION = 0.625


def display_duration(time_in_sec):
    remainder = time_in_sec % 60
    if remainder == time_in_sec:
        return f'{time_in_sec:.2f} seconds'
    else:
        seconds = remainder
        minutes = int(time_in_sec / 60)
        remainder = minutes % 60
        if remainder == minutes:
            return f'{minutes} mins, {seconds:.2f} seconds'
        else:
            hours   = int(minutes / 60)
            minutes = remainder
            remainder = hours % 24
            if remainder == hours:
                return f'{hours} hours, {minutes} mins, {seconds:.2f} seconds'
            else:
                days = int(hours / 24)
                hours = remainder
                return f'{days} days, {hours} hours, {minutes} mins, {seconds:.2f} seconds'


def parse_hour(hour_literal):
    try:
        result = HOUR_MAPPING[hour_literal]
        return result
    except KeyError:
        logging.error("unsupported hour ('%s')" % hour_literal)
        return None


def extract_header(line):
    match = CYCLONE_HEADER_PATTERN.match(line)
    if match:
        local_hurdat_id = match.group(1)
        local_nb_lines = match.group(2)
        return local_hurdat_id, int(local_nb_lines)
    else:
        raise Exception("unsupported header ('%s')" % line)


def extract_record(line, cyclone_id, hurdat2_id):
    match = CYCLONE_DESC_PATTERN.match(line)
    if match:
        record_id             = match.group(5).strip()
        status                = match.group(6)
        if not record_id and (status == 'HU' or status == 'TS'):
            year                = int(match.group(1))
            month               = int(match.group(2))
            day                 = int(match.group(3))
            hour_literal        = match.group(4)
            latitude_lit        = match.group(7)
            latitude_direction  = match.group(8)
            longitude_lit       = match.group(9)
            longitude_direction = match.group(10)
            max_sustained_wind  = int(match.group(11).strip())
            min_pressure        = int(match.group(12).strip())
            if max_sustained_wind < 0:
                max_sustained_wind = ''
            if min_pressure < 0:
                min_pressure = ''
            latitude = float(latitude_lit)
            longitude = float(longitude_lit)
            if latitude_direction == 'S':
                latitude = -latitude
            if longitude_direction == 'W':
                longitude = -longitude
            hour = parse_hour(hour_literal)
            if hour is None:
                return None
            values = [cyclone_id, hurdat2_id, year, month, day, hour, status,
                      latitude, longitude, max_sustained_wind, min_pressure]
            return values
        else:  # skip
            return None
    else:
        logging.error("unsupported record ('%s')" % line)
        return None


def main() -> None:
    # Default values
    dataset_parent_dir_path = '/data/sgardoll/cyclone_data/merra2_dataset'
    hurdat2_file_path = path.join(dataset_parent_dir_path,
                                  "hurdat2-1851-2019-052520.txt")

    if (len(sys.argv) > 1) and (sys.argv[1].strip()):
        hurdat2_file_path = sys.argv[1].strip()
        dataset_parent_dir_path = path.dirname(hurdat2_file_path)

    hurdat2_file = open(hurdat2_file_path, 'r')
    cyclone_list = []

    lines = hurdat2_file.readlines()
    cyclone_id = 0
    index = 0
    skipped_row_count = 0
    row_count = 0
    current_year  = -1
    previous_year = -1

    print(f'> starting to process {hurdat2_file_path}')
    while index < len(lines):
        current_line = lines[index]
        index = index + 1
        (hurdat_id, nb_lines) = extract_header(current_line)
        for index in range(index, (index + nb_lines)):
            current_line = lines[index]
            cyclone_values = extract_record(current_line, cyclone_id, hurdat_id)
            if cyclone_values is not None:
                cyclone_list.append(cyclone_values)
                row_count = row_count + 1
                current_year = cyclone_values[2]
                if current_year != previous_year:
                    print(f'  > processing year: {current_year}')
                    previous_year = current_year
            else:
                skipped_row_count = skipped_row_count + 1
        cyclone_id = cyclone_id + 1
        index = index + 1

    cyclone_dataframe = pd.DataFrame(data=cyclone_list, columns=CYCLONE_DF_COLUMNS)

    print(f'> number of row skipped: {skipped_row_count}')
    # 19 002

    hurdat2_file.close()

    print('> sorting cyclone dataset according to the date')
    cyclone_dataframe.sort_values(by=['year', 'month', 'day', 'hour'],
                                  ascending=True, inplace=True)

    print(f'> drop cyclones before {MERRA2_MIN_YEAR-1} included (MERRA-2 specs)')
    dropped_indexes = cyclone_dataframe[cyclone_dataframe['year'] < MERRA2_MIN_YEAR].index
    cyclone_dataframe.drop(dropped_indexes, inplace=True)

    print('> rebuilding the index of the cyclone dataset')
    cyclone_dataframe = cyclone_dataframe.reset_index(drop=True)

    #print('> translating into ERA5 coordinate systems')
    #import nxtensor.utils.coordinate_utils as cu
    #from nxtensor.utils.coordinates import CoordinateFormat
    #cu.reformat_coordinates(cyclone_dataframe, 'lat', CoordinateFormat.INCREASING_DEGREE_NORTH,
    #                        CoordinateFormat.DECREASING_DEGREE_NORTH, MERRA2_LAT_COORDINATES_RESOLUTION, 2)
    #cu.reformat_coordinates(cyclone_dataframe, 'lon', CoordinateFormat.M_180_TO_180_DEGREE_EAST,
    #                        CoordinateFormat.ZERO_TO_360_DEGREE_EAST, MERRA2_LON_COORDINATES_RESOLUTION, 2)

    filename = 'all_cyclone_dataset.csv'
    print(f'> saving {filename} ({len(cyclone_dataframe)} rows)')
    cyclone_dataframe_file_path = path.join(dataset_parent_dir_path, filename)
    cyclone_dataframe.to_csv(cyclone_dataframe_file_path, sep=',', na_rep='', header=True,
                             index=True, index_label='img_id', encoding='utf8', line_terminator='\n')
    # all_extraction.shape # (9684, 11)

    # Extraction of post 2000 records.
    extraction_2k = cyclone_dataframe[cyclone_dataframe['year'] >= 2000]
    filename = '2k_cyclone_dataset.csv'
    print(f'> saving {filename} ({len(extraction_2k)} rows)')
    extraction_2k_file_path = path.join(dataset_parent_dir_path, filename)
    extraction_2k.to_csv(extraction_2k_file_path, sep=',', na_rep='', header=True,
                          index=True, index_label='img_id', encoding='utf8', line_terminator='\n')

    # Extraction of year 2010.
    extraction_2010 = cyclone_dataframe[cyclone_dataframe.year == 2010]
    filename = '2010_cyclone_dataset.csv'
    print(f'> saving {filename} ({len(extraction_2010)} rows)')
    extraction_2010_file_path = path.join(dataset_parent_dir_path, filename)
    extraction_2010.to_csv(extraction_2010_file_path, sep=',', na_rep='', header=True,
                          index=True, index_label='img_id', encoding='utf8', line_terminator='\n')
    # extraction_2kb.shape # (5421, 11)


    stop = time.time()
    formatted_time = display_duration((stop-start))
    print(f'> spend {formatted_time} processing')


if __name__ == '__main__':
    main()
    sys.exit(0)

