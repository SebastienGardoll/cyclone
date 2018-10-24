#!/bin/bash

set -e
set -u

readonly BASE_DIR_PATH="$(pwd)"
SCRIPT_DIR_PATH="$(dirname $0)"; cd "${SCRIPT_DIR_PATH}"
readonly SCRIPT_DIR_PATH="$(pwd)" ; cd "${BASE_DIR_PATH}"

MINICONDA_HOME="${HOME}/miniconda2"
MINICONDA_ENV_PATH="${MINICONDA_HOME}/envs/sandbox"
source "${MINICONDA_HOME}/bin/activate" "${MINICONDA_ENV_PATH}"

readonly FILE_PREFIX='2k'
readonly CYCLONE_TENSOR_POSTFIX='cyclone_tensor'
readonly NO_CYCLONE_TENSOR_POSTFIX='no_cyclone_tensor'
readonly CYCLONE_DB_POSTFIX='extraction_dataset'
readonly NO_CYCLONE_DB_POSTFIX='no_cyclone_dataset'
readonly TENSOR_PARENT_DIR_PATH='/home/sgardoll/ouragan/tensors'

readonly MERGED_TENSOR_PARENT_DIR_PATH='/home/sgardoll/ouragan/merged_tensors'
readonly MERGED_PREFIX="merged_${FILE_PREFIX}"
readonly MERGED_TENSOR_POSTFIX='tensor'

# 0 means don't compute graphics for stats.
# 1 means compute graphics but don't display them.
# 2 means compute graphics and display them.
readonly GRAPHIC_MODE=1

date

cd "${SCRIPT_DIR_PATH}"

set +u
if [[ "${1}" = 'all' ]]; then
  echo -e "\n*********** BUILD CYCLONE DB ***********\n"
  python3 build_cyclone_db.py
else
  echo "> skip building cyclone db"
fi
set -u

echo -e "\n*********** BUILD NO CYCLONE DB ***********\n"
python3 build_no_cyclone_db.py "${FILE_PREFIX}"

echo -e "\n*********** BUILD CYCLONE TENSOR ***********\n"
python3 build_cyclone_tensor.py "${FILE_PREFIX}"

echo -e "\n*********** BUILD NO CYCLONE TENSOR ***********\n"
python3 build_no_cyclone_tensor.py "${FILE_PREFIX}"

echo -e "\n*********** MERGE TENSORS ***********\n"
python3 merge_tensors.py "${FILE_PREFIX}"

if [ ${GRAPHIC_MODE} -eq 0 ]; then
  echo "> skip computing stats"
else
  echo -e "\n*********** BUILD CYCLONE STATS ***********\n"
  python3 build_stats.py "${FILE_PREFIX}" "${CYCLONE_TENSOR_POSTFIX}" \
"${TENSOR_PARENT_DIR_PATH}" ${GRAPHIC_MODE}

  echo -e "\n*********** BUILD NO CYCLONE STATS ***********\n"
  python3 build_stats.py "${FILE_PREFIX}" "${NO_CYCLONE_TENSOR_POSTFIX}" \
"${TENSOR_PARENT_DIR_PATH}" ${GRAPHIC_MODE}
fi

echo -e "\n*********** BUILD MERGED TENSOR STATS ***********\n"
python3 build_stats.py "${MERGED_PREFIX}" "${MERGED_TENSOR_POSTFIX}" \
"${MERGED_TENSOR_PARENT_DIR_PATH}" ${GRAPHIC_MODE}

date
exit 0