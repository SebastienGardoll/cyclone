#!/bin/bash

                             ##### ENV VARS #####

# SCRIPT_DIR_PATH

                             ##### SETTINGS #####

export PYTHONUNBUFFERED='true'

                               ##### MAIN #####

set -e
set -u

echo "> month predict cyclone starting at $(date)"

echo "> sourcing common file"
source "${SCRIPT_DIR_PATH}/common.sh"

echo "> sourcing ${MINICONDA_ENV_PATH}"
source "${MINICONDA_HOME}/bin/activate" "${MINICONDA_ENV_PATH}"

cd "${SCRIPT_DIR_PATH}"

python3 month_predict_cyclone.py

echo "> month predict cyclone ending at $(date)"

exit 0
