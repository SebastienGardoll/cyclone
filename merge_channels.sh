#!/bin/bash

                             ##### ENV VARS #####

# Export these environement variables:
# NUM_CORE
# DATASET_PREFIX

# SCRIPT_DIR_PATH
                             ##### SETTINGS #####

                               ##### MAIN #####

set -e
set -u

echo "> merge channels starting at $(date)"

echo "> sourcing common file"
source "${SCRIPT_DIR_PATH}/common.sh"

source_conda_env

cd "${SCRIPT_DIR_PATH}"

python3 merge_channels.py "${DATASET_PREFIX}" ${NUM_CORE}

echo "> merge channels ending at $(date)"

exit 0
