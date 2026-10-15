#!/usr/bin/env bash
set -euo pipefail
# Standard 4-tuple contract; raw HDF5 root is explicit to avoid accidental dataset mixing.
bench_name=${1:?bench_name ckpt_name env_cfg_type action_type required}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
[[ "${action_type}" == "ee" ]] || { echo "Hy-VLA requires action_type=ee" >&2; exit 2; }
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_ROOT="${HY_VLA_ROOT:-${POLICY_DIR}/Hy-Embodied-0.5-VLA}"
: "${HDF5_DIR:?Set HDF5_DIR to the demonstration HDF5 tree}"
NORM_PATH="${NORM_PATH:-${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}/norm_stats.pkl}"
mkdir -p "$(dirname "${NORM_PATH}")"
cd "${HY_VLA_ROOT}"
python scripts/compute_norm_robodojo.py --hdf5-dir "${HDF5_DIR}" --output "${NORM_PATH}" \
  --downsample-rate "${HY_DOWNSAMPLE_RATE:-1}" --chunk-size "${HY_CHUNK_SIZE:-25}" --umi-coord-frame
