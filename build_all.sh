#!/bin/bash

set -e
set -u

source activate sandbox

date
FILE_PREFIX='2000'
CYCLONE_TENSOR_POSTFIX='cyclone_tensor'
NO_CYCLONE_TENSOR_POSTFIX='no_cyclone_tensor'
CYCLONE_DB_POSTFIX='extraction_dataset'
NO_CYCLONE_DB_POSTFIX='no_cyclone_dataset'
TENSOR_PARENT_DIR_PATH='/home/sgardoll/ouragan/tensors'

MERGED_TENSOR_PARENT_DIR_PATH='/home/sgardoll/ouragan/merged_tensors'
MERGED_PREFIX="merged_${FILE_PREFIX}"
MERGED_TENSOR_POSTFIX='tensor'

# 0 means don't compute graphics for stats.
# 1 means compute graphics but don't display them.
# 2 means compute graphics and display them.
GRAPHIC_MODE=1

echo "*********** BUILD CYCLONE DB ***********"
#python3 build_cyclone_db.py

echo "*********** BUILD NO CYCLONE DB ***********"
python3 build_no_cyclone_db.py "${FILE_PREFIX}"

echo "*********** BUILD CYCLONE TENSOR ***********"
python3 build_cyclone_tensor.py "${FILE_PREFIX}"

echo "*********** BUILD NO CYCLONE TENSOR ***********"
python3 build_no_cyclone_tensor.py "${FILE_PREFIX}"

echo "*********** BUILD CYCLONE STATS ***********"
python3 build_stats.py "${FILE_PREFIX}" "${CYCLONE_TENSOR_POSTFIX}"\
"${TENSOR_PARENT_DIR_PATH}" ${GRAPHIC_MODE}

echo "*********** BUILD NO CYCLONE STATS ***********"
python3 build_stats.py "${FILE_PREFIX}" "${NO_CYCLONE_TENSOR_POSTFIX}"\
"${TENSOR_PARENT_DIR_PATH}" ${GRAPHIC_MODE}


#echo "*********** BUILD MERGED TENSOR STATS ***********"
#python3 build_stats.py "${MERGED_PREFIX}" "${MERGED_TENSOR_POSTFIX}"\
"${MERGED_TENSOR_PARENT_DIR_PATH}" ${GRAPHIC_MODE}

date
exit 0
