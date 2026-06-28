#!/usr/bin/env bash
# Launch benchmark inference using the aim-graphcast conda environment.
# This single env supports ALL four models: GenCast, GraphCast, FourCastNet, Climatology.
#   JAX 0.4.30 + graphcast    →  GenCast, GraphCast
#   torch 2.6+cu124 + earth2mip  →  FourCastNet v2 + PrecipitationAFNO
#
# Usage:
#   ./run_inference.sh [benchmark-ea args...]
#
# Examples:
#   ./run_inference.sh --models climatology --start 2024-03-01 --end 2024-03-07
#   ./run_inference.sh --models gencast graphcast --start 2024-03-01 --end 2024-03-31
#   ./run_inference.sh --models fourcastnet --start 2024-03-01 --end 2024-03-31
#   ./run_inference.sh --models gencast graphcast fourcastnet climatology  # all models
#
#   # Save every variable (regridded to the EA grid):
#   ./run_inference.sh --models graphcast --save-variables all                       # ens mean for non-precip
#   ./run_inference.sh --models gencast --save-variables all --extra-var-members all # every member
#
# First-time: download FourCastNet v2 + PrecipAFNO weights (one-time, no API key):
#   PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/aim-graphcast/bin/python -c "
#     import sys; sys.path.insert(0, '.')
#     from earth2mip import registry
#     registry.get_model('e2mip://fcnv2_sm')
#     from earth2mip.diagnostic.precipitation_afno import PrecipitationAFNO
#     PrecipitationAFNO.load_package()"

set -euo pipefail

ENV=/home/ubuntu/miniconda3/envs/aim-graphcast
PYTHON=${ENV}/bin/python
SITE=${ENV}/lib/python3.10/site-packages

export PYTHONNOUSERSITE=1

export LD_LIBRARY_PATH=\
${SITE}/nvidia/cu13/lib:\
${SITE}/nvidia/cusparselt/lib:\
${SITE}/nvidia/cudnn/lib:\
${SITE}/nvidia/cublas/lib:\
${SITE}/nvidia/cuda_runtime/lib:\
${SITE}/nvidia/cuda_nvrtc/lib:\
${SITE}/nvidia/cufft/lib:\
${SITE}/nvidia/cusolver/lib:\
${SITE}/nvidia/cusparse/lib:\
${SITE}/nvidia/nvjitlink/lib:\
${SITE}/nvidia/nccl/lib:\
${ENV}/lib:\
${LD_LIBRARY_PATH:-}

export PYTHONWARNINGS="ignore::FutureWarning:google"

exec "${PYTHON}" -m benchmark_ea.run "$@"
