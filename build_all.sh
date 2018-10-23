#!/bin/bash

date
FILE_PREFIX='2000_10'
CYCLONE_TENSOR_POSTFIX='cyclone_tensor.npy'
NO_CYCLONE_TENSOR_POSTFIX='no_cyclone_tensor.npy'
CYCLONE_DB_POSTFIX='extraction_dataset.csv'
NO_CYCLONE_DB_POSTFIX='no_cyclone_tensor.csv'

#python3 build_cyclone_db.py
python3 build_no_cyclone_db.py "${FILE_PREFIX}"
python3 build_cyclone_tensor.py "${FILE_PREFIX}"
python3 build_no_cyclone_tensor.py "${FILE_PREFIX}"
python3 build_stats.py "${FILE_PREFIX}" "${CYCLONE_TENSOR_POSTFIX}" "${CYCLONE_DB_POSTFIX}"
python3 build_stats.py "${FILE_PREFIX}" "${NO_CYCLONE_TENSOR_POSTFIX}" "${NO_CYCLONE_DB_POSTFIX}"
date