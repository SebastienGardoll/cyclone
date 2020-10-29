#!/bin/bash

                                  ##### SETTINGS #####

SCRIPT_DIR_PATH="$(dirname $0)" ; cd "${SCRIPT_DIR_PATH}" ; readonly SCRIPT_DIR_PATH="$(pwd)"

# Python:
export PYTHONUNBUFFERED='true'
ERROR_EXIT_CODE=1
set -e

                                  ##### FUNCTIONS #####

function source_conda_env
{
  local c_home="${1}"
  local c_env_name="${2}"
  source "${c_home}/bin/activate" "${c_env_name}"
}

                                     ##### MAIN #####

if [[ -z "${1}" ]]; then
  echo "> [ERROR] missing conda home path. Abort."
  exit ${ERROR_EXIT_CODE}
fi

if [[ -z "${2}" ]]; then
  echo "> [ERROR] missing conda environment name. Abort."
  exit ${ERROR_EXIT_CODE}
fi

if [[ -z "${3}" ]]; then
  echo "> [ERROR] missing python script path. Abort."
  exit ${ERROR_EXIT_CODE}
fi

readonly conda_home="${1}"
readonly conda_env_name="${2}"
readonly python_script_path="${3}"
readonly python_script_name="$(basename ${python_script_path})"

echo "> source conda env: ${conda_env_name}"
source_conda_env ${conda_home} ${conda_env_name}

#echo "> loading python3 module"
#module load python/3.7-anaconda2019-10

echo "> starting ${python_script_name} ($(date))"
cd "${SCRIPT_DIR_PATH}"
/net/nfs/tools/anaconda/3.7/bin/python3 "${python_script_path}"

echo "> ${python_script_name} is completed with return code: ${?} ($(date))"
exit 0
