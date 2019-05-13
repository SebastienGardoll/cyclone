#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:36:59 2018

@author: Sébastien Gardoll
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

HOUR_MAPPING = {'0000':0, '0600':6, '1200':12, '1800':18}

CYCLONE_DF_COLUMNS      = ['cyclone_id', 'hurdat2_id', 'year', 'month', 'day',
                           'hour', 'status', 'lat', 'lon',\
                           'max_sustained_wind', 'min_pressure']

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
    logging.error("unsupported hour ('%s')"%(hour_literal))
    return None


def extract_header(line):
  match = CYCLONE_HEADER_PATTERN.match(line)
  if match:
    hurdat_id = match.group(1)
    nb_lines = match.group(2)
    return (hurdat_id, int(nb_lines))
  else:
    raise Exception("unsupported header ('%s')"%(line))


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
      values = [cyclone_id, hurdat2_id, year, month, day, hour, status,\
                latitude, longitude, max_sustained_wind, min_pressure]
      return values
    else: # skip
      return None
  else:
    logging.error("unsupported record ('%s')"%(line))
    return None


# Default values
dataset_parent_dir_path = '/data/sgardoll/cyclone_data/dataset'
hurdat2_file_path = path.join(dataset_parent_dir_path,
                              "hurdat2-1851-2017-050118.txt")

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  hurdat2_file_path = sys.argv[1].strip()
  dataset_parent_dir_path = path.dirname(hurdat2_file_path)

# hurdat2_file_path = '/home/sgardoll/cyclone/dataset/tmp' # DEBUG
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
  first_record_line = row_count
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
  last_record_line_plus_1 = row_count
  mapping_values = [cyclone_id, hurdat_id, first_record_line, last_record_line_plus_1]
  cyclone_id = cyclone_id + 1
  index = index + 1

cyclone_dataframe = pd.DataFrame(data=cyclone_list, columns=CYCLONE_DF_COLUMNS)

print(f'> number of row skipped: {skipped_row_count}')
# 18 198

hurdat2_file.close()

print('> sorting cyclone dataset according to the date')
cyclone_dataframe.sort_values(by=['year', 'month', 'day', 'hour'],
                              ascending=True, inplace=True)

print('> rebuilding the index of the cyclone dataset')
cyclone_dataframe = cyclone_dataframe.reset_index(drop=True)

filename = 'all_cyclone_dataset.csv'
print(f'> saving {filename} ({len(cyclone_dataframe)} rows)')
cyclone_dataframe_file_path = path.join(dataset_parent_dir_path, filename)
cyclone_dataframe.to_csv(cyclone_dataframe_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')

# Extraction of post 2000 records.
extraction_2ka = cyclone_dataframe[cyclone_dataframe['year'] >= 2000]
filename = '2ka_cyclone_dataset.csv'
print(f'> saving {filename} ({len(extraction_2ka)} rows)')
extraction_2ka_file_path = path.join(dataset_parent_dir_path, filename)
extraction_2ka.to_csv(extraction_2ka_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
#extraction_2k.shape # (4952, 11)

# Extraction of post 2000 records except August of 2000.
extraction_2kb = cyclone_dataframe[(cyclone_dataframe.year >= 2000) &
     (~ ((cyclone_dataframe.year == 2000) & (cyclone_dataframe.month == 8)))]
filename = '2kb_cyclone_dataset.csv'
print(f'> saving {filename} ({len(extraction_2kb)} rows)')
extraction_2kb_file_path = path.join(dataset_parent_dir_path, filename)
extraction_2kb.to_csv(extraction_2kb_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
#extraction_2kb.shape # (4853, 11)

year_extraction = extraction_2ka[extraction_2ka['year'] < 2001]
filename = '2000_cyclone_dataset.csv'
print(f'> saving {filename} ({len(year_extraction)} rows)')
year_extraction_file_path = path.join(dataset_parent_dir_path, filename)
year_extraction.to_csv(year_extraction_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
#year_extraction.shape # (265, 11)

month_extraction = year_extraction[year_extraction['month'] == 10]
filename = '2000_10_cyclone_dataset.csv'
print(f'> saving {filename} ({len(month_extraction)} rows)')
month_extraction_file_path = path.join(dataset_parent_dir_path, filename)
month_extraction.to_csv(month_extraction_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
#month_extraction.shape # (49, 11)

stop = time.time()
formatted_time =display_duration((stop-start))
print(f'> spend {formatted_time} processing')
