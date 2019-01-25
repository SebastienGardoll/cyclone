#!/bin/bash

                             ##### ENV VARS #####

# Export these environement variables:
# CHANNEL_NAME
# NUM_CORE
# DATASET_PREFIX

# SCRIPT_DIR_PATH
                             ##### SETTINGS #####

export PYTHONUNBUFFERED='true'

                               ##### MAIN #####

set -e
set -u

echo "> build channels starting at $(date)"

echo "> sourcing common file"
source "${SCRIPT_DIR_PATH}/common.sh"

echo "> sourcing ${MINICONDA_ENV_PATH}"
source "${MINICONDA_HOME}/bin/activate" "${MINICONDA_ENV_PATH}"

cd "${SCRIPT_DIR_PATH}"

python3 build_channels.py "${DATASET_PREFIX}" "${CHANNEL_NAME}" ${NUM_CORE}

echo "> build channels ending at $(date)"

exit 0