#!/bin/bash

set -e
set -u

echo "> starting at $(date)"

readonly BASE_DIR_PATH="$(pwd)"
SCRIPT_DIR_PATH="$(dirname $0)"; cd "${SCRIPT_DIR_PATH}"
readonly SCRIPT_DIR_PATH="$(pwd)"

echo "> source common file"
source "${SCRIPT_DIR_PATH}/common.sh"

source_conda_env

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

echo -e "\n*********** BUILD CONV NET ***********\n"

cd "${SCRIPT_DIR_PATH}"

python3 train_conv_net.py "${FILE_PREFIX}" ${NUM_THREADS}

echo "> ending at $(date)"

exit 0
