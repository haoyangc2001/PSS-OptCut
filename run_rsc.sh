#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="/home/caohy/miniconda3"
CONDA_ENV="RSC"
PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
GUROBI_HOME="/home/caohy/app/gurobi/gurobi1301/linux64"
GRB_LICENSE_FILE="/home/caohy/opt/gurobi/gurobi.lic"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ ! -x "${GUROBI_HOME}/bin/gurobi_cl" ]]; then
  echo "gurobi_cl not found: ${GUROBI_HOME}/bin/gurobi_cl" >&2
  exit 1
fi

if [[ ! -f "${GRB_LICENSE_FILE}" ]]; then
  echo "Gurobi license not found: ${GRB_LICENSE_FILE}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export GUROBI_HOME
export GRB_LICENSE_FILE
export PATH="${GUROBI_HOME}/bin:${CONDA_ROOT}/bin:${PATH}"
# Avoid inheriting incompatible libraries from other conda environments.
export LD_LIBRARY_PATH="${GUROBI_HOME}/lib"
unset PYTHONPATH

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "Python: ${PYTHON_BIN}"
echo "GUROBI_HOME: ${GUROBI_HOME}"
echo "GRB_LICENSE_FILE: ${GRB_LICENSE_FILE}"

exec "${PYTHON_BIN}" src/main.py "$@"
