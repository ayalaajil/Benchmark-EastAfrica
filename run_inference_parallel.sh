#!/usr/bin/env bash
# Run inference for all models in parallel, one GPU each, over any
# start/end date range 

# GPU assignments (change if those GPUs are busy):
#   GPU 0 → GenCast      (JAX, ~40 GB for 50 members)
#   GPU 1 → GraphCast    (JAX, ~15 GB deterministic)
#   GPU 2 → FourCastNet  (PyTorch, ~20 GB FCNv2 + PrecipAFNO)
#
# Each model's output goes to ${OUTPUT_DIR}/<model>/ (default data/predictions).
# Logs: logs/<start-year>/<model>.log (e.g. logs/2024/, logs/2025/ — override
# with LOGDIR to use a custom label, e.g. for a season-specific run).
#
# Usage:
#   ./run_inference_parallel.sh                      # default period (2024), all 3 models, precip only
#   ./run_inference_parallel.sh gencast fourcastnet  # subset
#
#   # A different year or season — START/END are never hardcoded:
#   START=2025-01-01 END=2025-12-24 ./run_inference_parallel.sh
#   START=2024-10-01 END=2024-12-24 LOGDIR=logs/2024_OND ./run_inference_parallel.sh
#
#   # Save every variable (ensemble mean for non-precip fields):
#   SAVE_VARIABLES=all ./run_inference_parallel.sh
#
    # Save every variable for every ensemble member (large):
#   SAVE_VARIABLES=all EXTRA_VAR_MEMBERS=all ./run_inference_parallel.sh
#
#   # Full 0.25° checkpoints (gencast/graphcast only — fourcastnet is already
#   # native 0.25° and ignores this flag) into a SEPARATE output dir, since
#   # the default OUTPUT_DIR would otherwise overwrite the 1° predictions:
#   RESOLUTION=0.25 OUTPUT_DIR=/mnt/vol800/predictions_0p25 \
#       ./run_inference_parallel.sh gencast graphcast
#
# Tunable via environment: START END LEAD_DAYS SAVE_VARIABLES EXTRA_VAR_MEMBERS
#                          N_MEMBERS OUTPUT_DIR RESOLUTION LOGDIR


set -euo pipefail

START="${START:-2024-01-01}"
END="${END:-2024-12-24}"
LEAD_DAYS="${LEAD_DAYS:-1 3 5 7}"
SAVE_VARIABLES="${SAVE_VARIABLES:-all}"      # precip | all
EXTRA_VAR_MEMBERS="${EXTRA_VAR_MEMBERS:-mean}"  # mean | all  (only for save=all)
N_MEMBERS="${N_MEMBERS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-data/predictions}"
RESOLUTION="${RESOLUTION:-1.0}"   # "1.0" (small/mini, default) | "0.25" (flagship);
                                  # only gencast/graphcast use this — ignored by
                                  # fourcastnet (already native 0.25°)

# GPU index per model — adjust if those GPUs are occupied
declare -A GPU_FOR=(
    [gencast]=0
    [graphcast]=1
    [fourcastnet]=2
)

ALL_MODELS=(gencast graphcast fourcastnet)
MODELS=("${@:-${ALL_MODELS[@]}}")   # use args if given, else all models

ENV=/home/ubuntu/miniconda3/envs/aim-graphcast
PYTHON=${ENV}/bin/python
SITE=${ENV}/lib/python3.10/site-packages

# Defaults to the start date's year (logs/2024, logs/2025, …) so different
# runs don't collide; override LOGDIR directly for a custom label (e.g. a
# season-specific run in the same year).
LOGDIR="${LOGDIR:-logs/${START:0:4}}"
mkdir -p "${LOGDIR}"

# Common env vars (exported so subshells inherit them)
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

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp)  Starting benchmark inference  [${START} → ${END}]  (parallel)"
echo "$(timestamp)  Models: ${MODELS[*]}"
echo "$(timestamp)  Resolution: ${RESOLUTION}°  (ignored by models that aren't resolution-aware)"
echo "$(timestamp)  Output: ${OUTPUT_DIR}"
echo "$(timestamp)  Logs:   ${LOGDIR}/"
echo ""

PIDS=()
MODEL_NAMES=()
OVERALL_START=$(date +%s)

for model in "${MODELS[@]}"; do
    if [[ -z "${GPU_FOR[$model]+x}" ]]; then
        echo "$(timestamp)  [WARN] No GPU assigned for '${model}', skipping"
        continue
    fi

    gpu="${GPU_FOR[$model]}"
    log="${LOGDIR}/${model}.log"

    echo "$(timestamp)  Launching ${model^^} on GPU ${gpu}  →  ${log}"

    CUDA_VISIBLE_DEVICES=${gpu} \
        "${PYTHON}" -m benchmark_ea.run \
            --models "${model}" \
            --resolution "${RESOLUTION}" \
            --start  "${START}" \
            --end    "${END}" \
            --lead-days ${LEAD_DAYS} \
            --save-variables "${SAVE_VARIABLES}" \
            --extra-var-members "${EXTRA_VAR_MEMBERS}" \
            --n-members "${N_MEMBERS}" \
            --output-dir "${OUTPUT_DIR}" \
        > "${log}" 2>&1 &

    PIDS+=($!)
    MODEL_NAMES+=("${model}")
done

echo ""
echo "$(timestamp)  All ${#PIDS[@]} jobs launched.  Waiting for completion …"
echo "$(timestamp)  Monitor progress:"
for model in "${MODEL_NAMES[@]}"; do
    echo "    tail -f ${LOGDIR}/${model}.log"
done
echo ""

# Wait for each job and report exit status
ALL_OK=true
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    model="${MODEL_NAMES[$i]}"
    if wait "${pid}"; then
        echo "$(timestamp)  [OK]   ${model^^} finished"
    else
        echo "$(timestamp)  [FAIL] ${model^^} exited with error (see ${LOGDIR}/${model}.log)"
        ALL_OK=false
    fi
done

OVERALL_END=$(date +%s)
TOTAL=$(( OVERALL_END - OVERALL_START ))
echo ""
echo "$(timestamp)  Done  ($(( TOTAL / 60 ))m $(( TOTAL % 60 ))s total)"

if $ALL_OK; then
    echo "$(timestamp)  All models completed successfully."
else
    echo "$(timestamp)  One or more models failed — check logs above."
    exit 1
fi
