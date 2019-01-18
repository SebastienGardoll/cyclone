#!/bin/bash

                             ##### SETTINGS #####

# Bash:

set -e
set -u

# Python:

export PYTHONUNBUFFERED='true'

# Paths:

readonly BASE_DIR_PATH="$(pwd)"
SCRIPT_DIR_PATH="$(dirname $0)"; cd "${SCRIPT_DIR_PATH}"
readonly SCRIPT_DIR_PATH="$(pwd)"

readonly DATA_DIR_PATH='/data/sgardoll/cyclone_data'
readonly DATA_BACKUP_DIR_PATH='/data/sgardoll/cyclone_data.clean'
readonly JOB_LOG_DIR_PATH="${SCRIPT_DIR_PATH}/jobs"

# default job values:

readonly DEFAULT_DATASET_PREFIX='2000_10'
readonly DEFAULT_BUILD_KIND='skip'
readonly DEFAULT_MAX_WALL_TIME='01:59:59'
readonly DEFAULT_NUM_CORE=4
readonly DEFAULT_JOB_MEM='10gb'

# channel jobs:

readonly VARIABLE_NAMES=( 'MSL' 'TCWV' 'V10' 'U10' 'TA200' 'TA500' 'U850' 'V850' )
readonly CHANNEL_JOB_NUM_CORE=${DEFAULT_NUM_CORE}
readonly CHANNEL_JOB_MAX_WALL_TIME="${DEFAULT_MAX_WALL_TIME}"
readonly CHANNEL_JOB_MEM="${DEFAULT_JOB_MEM}"

# merge job:

readonly MERGE_JOB_NUM_CORE=${DEFAULT_NUM_CORE}
readonly MERGE_JOB_MAX_WALL_TIME="${DEFAULT_MAX_WALL_TIME}"
readonly MERGE_JOB_MEM="${DEFAULT_JOB_MEM}"

                               ##### FUNCTIONS #####

# common.sh must be sourced.
function source_conda_env
{
  echo "> source conda env: ${MINICONDA_ENV_PATH}"
  source "${MINICONDA_HOME}/bin/activate" "${MINICONDA_ENV_PATH}"
}

                               ##### MAIN #####

echo "> build tensor starting at $(date)"

echo "> source common file"

source "${SCRIPT_DIR_PATH}/common.sh"

set +u

ARGS_FIRST='false'

if [[ -n "${1}" ]]; then
  readonly DATASET_PREFIX="${1}"
  ARGS_FIRST='true'
else
  readonly DATASET_PREFIX="${DEFAULT_DATASET_PREFIX}"
fi

if [[ "${ARGS_FIRST}" == 'true' ]]; then
  if [[ -n "${2}" ]]; then
    readonly BUILD_KIND="${2}"
  else
    echo '#### missing build option ####'
    exit 1
  fi
else
  readonly BUILD_KIND="${DEFAULT_BUILD_KIND}"
fi

set -u

cd "${SCRIPT_DIR_PATH}"

if [[ "${BUILD_KIND}" = 'skip' ]]; then
  echo "> skip building dbs"
fi

if [[ "${BUILD_KIND}" = 'very_all' ]]; then

  echo -e "\n*********** CLEAN ${DATA_DIR_PATH} ***********\n"

  rm -vfr "${DATA_DIR_PATH}"
  cp -vrp "${DATA_BACKUP_DIR_PATH}" "${DATA_DIR_PATH}"

  echo -e "\n*********** BUILD CYCLONE DB ***********\n"
  source_conda_env
  python3 build_cyclone_db.py

  echo -e "\n*********** BUILD NO CYCLONE DB ***********\n"
  python3 build_no_cyclone_db.py "${DATASET_PREFIX}"
fi

if [[ "${BUILD_KIND}" = 'all' ]]; then
  echo -e "\n*********** BUILD NO CYCLONE DB ***********\n"
  source_conda_env
  python3 build_no_cyclone_db.py "${DATASET_PREFIX}"
fi

mkdir -p ${JOB_LOG_DIR_PATH}

echo -e "\n*********** BUILD CHANNELS ***********\n"

for index in ${!VARIABLE_NAMES[*]}
do
  current_channel="${VARIABLE_NAMES[index]}"
  echo "> starting build channel ${current_channel} job"
  job_num=$(qsub -o "${JOB_LOG_DIR_PATH}" -N "build_channels_${current_channel}" \
-v DATASET_PREFIX="${DATASET_PREFIX}",NUM_CORE=${CHANNEL_JOB_NUM_CORE},\
CHANNEL_NAME="${current_channel}",SCRIPT_DIR_PATH="${SCRIPT_DIR_PATH}" \
-j oe -l walltime="${CHANNEL_JOB_MAX_WALL_TIME}" \
-l mem=${CHANNEL_JOB_MEM} -l vmem=${CHANNEL_JOB_MEM} \
-l nodes=1:ppn=${CHANNEL_JOB_NUM_CORE} "${SCRIPT_DIR_PATH}/build_channels.sh")
job_nums[index]=${job_num}
  echo "  > job number is ${job_num}"
done

echo -e "\n*********** BUILD TENSOR ***********\n"

echo "> starting merge channels job"

dependency_list='depend=afterok'
for index in ${!job_nums[*]}
do
  dependency_list="${dependency_list}:${job_nums[index]}"
done

job_num=$(qsub -o "${JOB_LOG_DIR_PATH}" \
-v DATASET_PREFIX="${DATASET_PREFIX}",NUM_CORE=${MERGE_JOB_NUM_CORE},\
SCRIPT_DIR_PATH="${SCRIPT_DIR_PATH}" \
-m ae -j oe -l walltime="${MERGE_JOB_MAX_WALL_TIME}" \
-l mem=${MERGE_JOB_MEM} -l vmem=${MERGE_JOB_MEM} \
-W "${dependency_list}" \
-l nodes=1:ppn=${MERGE_JOB_NUM_CORE} "${SCRIPT_DIR_PATH}/merge_channels.sh")

echo "  > job number is ${job_num}"

echo -e "\n\n> build tensor completed at $(date)"

exit 0
