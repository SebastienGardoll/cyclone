#!/bin/bash

date
FILE_PREFIX='2000_10'
CYCLONE_TENSOR_POSTFIX='cyclone_tensor.npy'
NO_CYCLONE_TENSOR_POSTFIX='no_cyclone_tensor.npy'
CYCLONE_DB_POSTFIX='extraction_dataset.csv'
NO_CYCLONE_DB_POSTFIX='no_cyclone_dataset.csv'

echo "*********** BUILD CYCLONE DB ***********"
python3 build_cyclone_db.py

echo "*********** BUILD NO CYCLONE DB ***********"
python3 build_no_cyclone_db.py "${FILE_PREFIX}"

echo "*********** BUILD CYCLONE TENSOR ***********"
python3 build_cyclone_tensor.py "${FILE_PREFIX}"

echo "*********** BUILD NO CYCLONE TENSOR ***********"
python3 build_no_cyclone_tensor.py "${FILE_PREFIX}"

echo "*********** BUILD CYCLONE STATS ***********"
python3 build_stats.py "${FILE_PREFIX}" "${CYCLONE_TENSOR_POSTFIX}" "${CYCLONE_DB_POSTFIX}"

echo "*********** BUILD NO CYCLONE STATS ***********"
python3 build_stats.py "${FILE_PREFIX}" "${NO_CYCLONE_TENSOR_POSTFIX}" "${NO_CYCLONE_DB_POSTFIX}"
date