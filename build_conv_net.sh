#!/bin/bash

set -e
set -u

echo "> starting at $(date)"

MINICONDA_HOME="/data/sgardoll/miniconda2"
MINICONDA_ENV_PATH="${MINICONDA_HOME}/envs/sandbox"
echo "> sourcing ${MINICONDA_ENV_PATH}"
source "${MINICONDA_HOME}/bin/activate" "${MINICONDA_ENV_PATH}"

set +u

ARGS_FIRST='false'

if [[ -n "${1}" ]]; then
  readonly FILE_PREFIX="${1}"
  ARGS_FIRST='true'
else
  if [[ -n "${DATASET_PREFIX}" ]]; then
    readonly FILE_PREFIX="${DATASET_PREFIX}"
  else
    readonly FILE_PREFIX="2000_10"
  fi
fi

if [[ "${ARGS_FIRST}" == 'true' ]]; then
  if [[ -n "${2}" ]]; then
    readonly NUM_THREADS=${2}
  else
    echo '#### missing core number option ####'
    exit 1
  fi
else
  if [[ -n "${NUM_CORE}" ]]; then
    readonly NUM_THREADS=${NUM_CORE}
  else
    readonly NUM_THREADS=1
  fi
fi

set -u

export PYTHONUNBUFFERED='true'

readonly SCRIPT_DIR_PATH='/home/sgardoll/cyclone/spyder'

echo -e "\n*********** BUILD CONV NET ***********\n"

cd "${SCRIPT_DIR_PATH}"

python3 train_conv_net.py "${FILE_PREFIX}" ${NUM_THREADS}

echo "> ending at $(date)"

exit 0
