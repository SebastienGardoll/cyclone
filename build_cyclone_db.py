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

import common

CYCLONE_HEADER_PATTERN = re.compile('^([A-Z0-9]+), +[-\w]+, +(\d+),$')
CYCLONE_DESC_PATTERN = re.compile('^(\d{4})(\d{2})(\d{2}), (\d{4}), ([ A-Z]), ([A-Z]{2}), +(\d+\.\d+)([NS]), +(\d+\.\d+)([WE]), +(-?[\d ]+), +(-?[\d ]+).+$')

HOUR_TIME_STEP_MAPPING = {'0000':0, '0600':1, '1200':2, '1800':3}

CYCLONE_DF_COLUMNS      = ['cyclone_id', 'hurdat2_id', 'year', 'month', 'day',\
                           'time_step', 'status', 'lat', 'lon',\
                           'max_sustained_wind', 'min_pressure']
CYCLONE_MAPPING_COLUMNS = ['cyclone_id', 'hurdat2_id', 'first_img_id', 'last_img_id_plus_1']

def parse_hour(hour_literal):
  try:
    result = HOUR_TIME_STEP_MAPPING[hour_literal]
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
      time_step = parse_hour(hour_literal)
      if time_step is None:
        return None
      values = [cyclone_id, hurdat2_id, year, month, day, time_step, status,\
                latitude, longitude, max_sustained_wind, min_pressure]
      return values
    else: # skip
      return None
  else:
    logging.error("unsupported record ('%s')"%(line))
    return None

# Default values
dataset_parent_dir_path=common.DATASET_PARENT_DIR_PATH
hurdat2_file_path = path.join(dataset_parent_dir_path, "hurdat2-1851-2017-050118.txt")

if (len(sys.argv) > 1) and (sys.argv[1].strip()):
  hurdat2_file_path = sys.argv[1].strip()
  dataset_parent_dir_path = path.dirname(hurdat2_file_path)

#hurdat2_file_path = '/home/sgardoll/ouragan/dataset/tmp' # DEBUG
hurdat2_file = open(hurdat2_file_path, 'r')
cyclone_dataframe = pd.DataFrame(columns=CYCLONE_DF_COLUMNS)
cyclone_mapping   = pd.DataFrame(columns=CYCLONE_MAPPING_COLUMNS)

lines = hurdat2_file.readlines()
cyclone_id = 0
index = 0
skipped_row_count = 0
row_count = 0

while index < len(lines):
  current_line = lines[index]
  index = index + 1
  (hurdat_id, nb_lines) = extract_header(current_line)
  first_record_line = row_count
  for index in range(index, (index + nb_lines)):
    current_line = lines[index]
    cyclone_values = extract_record(current_line, cyclone_id, hurdat_id)
    if cyclone_values is not None:
      cyclone_row = pd.Series(cyclone_values, index=CYCLONE_DF_COLUMNS)
      cyclone_dataframe = cyclone_dataframe.append(cyclone_row, ignore_index=True)
      row_count = row_count + 1
    else:
      skipped_row_count = skipped_row_count + 1
  last_record_line_plus_1 = row_count
  mapping_values = [cyclone_id, hurdat_id, first_record_line, last_record_line_plus_1]
  cyclone_mapping_row = pd.Series(mapping_values, index=CYCLONE_MAPPING_COLUMNS)
  cyclone_mapping = cyclone_mapping.append(cyclone_mapping_row, ignore_index=True)
  cyclone_id = cyclone_id + 1
  index = index + 1

print(skipped_row_count)
# 18 198

hurdat2_file.close()

cyclone_mapping_file_path = path.join(dataset_parent_dir_path, "cyclone_mapping.csv")
cyclone_mapping.to_csv(cyclone_mapping_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='id', encoding = 'utf8', line_terminator = '\n')

cyclone_dataframe_file_path = path.join(dataset_parent_dir_path, "cyclone_dataset.csv")
cyclone_dataframe.to_csv(cyclone_dataframe_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')

# Extraction of post 2000 records.
extraction_2k = cyclone_dataframe[cyclone_dataframe['year'] >= 2000]
extraction_2k_file_path = path.join(dataset_parent_dir_path, "2k_extraction_dataset.csv")
extraction_2k.to_csv(extraction_2k_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
extraction_2k.shape # (4952, 11)

year_extraction = extraction_2k[extraction_2k['year'] < 2001]
year_extraction_file_path = path.join(dataset_parent_dir_path, "2000_extraction_dataset.csv")
year_extraction.to_csv(year_extraction_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
year_extraction.shape # (265, 11)

month_extraction = year_extraction[year_extraction['month'] == 10]
month_extraction_file_path = path.join(dataset_parent_dir_path, "2000_10_extraction_dataset.csv")
month_extraction.to_csv(month_extraction_file_path, sep = ',', na_rep = '', header = True,\
  index = True, index_label='img_id', encoding = 'utf8', line_terminator = '\n')
month_extraction.shape # (49, 11)

exit(0)