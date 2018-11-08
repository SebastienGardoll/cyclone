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

readonly CYCLONE_CHANNEL_POSTFIX='cyclone_channel'
readonly NO_CYCLONE_CHANNEL_POSTFIX='no_cyclone_channel'
readonly CYCLONE_DB_POSTFIX='extraction_dataset'
readonly NO_CYCLONE_DB_POSTFIX='no_cyclone_dataset'

readonly MERGED_PREFIX="merged_${FILE_PREFIX}"
readonly MERGED_CHANNEL_POSTFIX='channel'

readonly PROJECT_DIR_PATH='/home/sgardoll/ouragan'
readonly TENSOR_PARENT_DIR_PATH="${PROJECT_DIR_PATH}/tensor"
readonly MERGED_CHANNEL_PARENT_DIR_PATH="${PROJECT_DIR_PATH}/merged_channels"
readonly CHANNEL_PARENT_DIR_PATH="${PROJECT_DIR_PATH}/channels"

# 0 means don't compute graphics for stats.
# 1 means compute graphics but don't display them.
# 2 means compute graphics and display them.
readonly GRAPHIC_MODE=1

readonly NUM_PROCESSES=8

date

cd "${SCRIPT_DIR_PATH}"

set +u

if [[ -z "${1}" ]]; then
  echo "> skip building dbs"
fi

if [[ "${1}" = 'very_all' ]]; then
  echo -e "\n*********** BUILD CYCLONE DB ***********\n"
  python3 build_cyclone_db.py

  echo -e "\n*********** BUILD NO CYCLONE DB ***********\n"
  python3 build_no_cyclone_db.py "${FILE_PREFIX}"
fi

if [[ "${1}" = 'all' ]]; then
  echo -e "\n*********** BUILD NO CYCLONE DB ***********\n"
  python3 build_no_cyclone_db.py "${FILE_PREFIX}"
fi

set -u

echo -e "\n*********** BUILD CYCLONE CHANNELS ***********\n"
python3 build_cyclone_channels.py "${FILE_PREFIX}"

echo -e "\n*********** BUILD NO CYCLONE CHANNELS ***********\n"
python3 build_no_cyclone_channels.py "${FILE_PREFIX}"

echo -e "\n*********** MERGE CHANNELS ***********\n"
python3 merge_channels.py "${FILE_PREFIX}" ${NUM_PROCESSES}

if [ ${GRAPHIC_MODE} -eq 0 ]; then
  echo "> skip computing stats"
else
  echo -e "\n*********** BUILD CYCLONE STATS ***********\n"
  python3 build_stats.py "${FILE_PREFIX}" "${CYCLONE_CHANNEL_POSTFIX}" \
"${CHANNEL_PARENT_DIR_PATH}" ${GRAPHIC_MODE}

  echo -e "\n*********** BUILD NO CYCLONE STATS ***********\n"
  python3 build_stats.py "${FILE_PREFIX}" "${NO_CYCLONE_CHANNEL_POSTFIX}" \
"${CHANNEL_PARENT_DIR_PATH}" ${GRAPHIC_MODE}
fi

date
exit 0
