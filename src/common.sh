                              #### GLOBAL VARS ####

readonly MINICONDA_HOME="/data/sgardoll/miniconda2"
readonly MINICONDA_ENV_PATH="${MINICONDA_HOME}/envs/sandbox"

readonly DATA_DIR_PATH='/data/sgardoll/cyclone_data'
readonly DATA_BACKUP_DIR_PATH='/data/sgardoll/cyclone_data.clean'
readonly JOB_LOG_DIR_PATH="${SCRIPT_DIR_PATH}/jobs"

# Python:

export PYTHONUNBUFFERED='true'

                               ##### FUNCTIONS #####

function source_conda_env
{
  echo "> source conda env: ${MINICONDA_ENV_PATH}"
  source "${MINICONDA_HOME}/bin/activate" "${MINICONDA_ENV_PATH}"
}