#!/usr/bin/env bash
# Run MAM 2024 inference for all models in parallel, one GPU each.
#
# GPU assignments (change if those GPUs are busy):
#   GPU 0 → GenCast      (JAX, ~40 GB for 50 members)
#   GPU 1 → GraphCast    (JAX, ~15 GB deterministic)
#   GPU 2 → FourCastNet  (PyTorch, ~20 GB FCNv2 + PrecipAFNO)
#
# Each model's output goes to ${OUTPUT_DIR}/<model>/ (default data/predictions).
# Logs: logs/2024/<model>.log
#
# Usage:
#   ./run_inference_parallel.sh                      # all 3 models, precip only
#   ./run_inference_parallel.sh gencast fourcastnet  # subset
#
#   # Save every variable (ensemble mean for non-precip fields):
#   SAVE_VARIABLES=all ./run_inference_parallel.sh
#   
    # Save every variable for every ensemble member (large):
#   SAVE_VARIABLES=all EXTRA_VAR_MEMBERS=all ./run_inference_parallel.sh
#
# Tunable via environment: START END LEAD_DAYS SAVE_VARIABLES EXTRA_VAR_MEMBERS
#                          N_MEMBERS OUTPUT_DIR
#
# Background (detach from terminal):
#   nohup ./run_inference_parallel.sh > logs/2024/parallel.log 2>&1 &

set -euo pipefail

# Inference window and saving options (override via environment, e.g.
#   SAVE_VARIABLES=all EXTRA_VAR_MEMBERS=mean ./run_inference_parallel.sh)


START="${START:-2024-01-01}"
END="${END:-2024-12-24}"
LEAD_DAYS="${LEAD_DAYS:-1 3 5 7}"
SAVE_VARIABLES="${SAVE_VARIABLES:-all}"      # precip | all
EXTRA_VAR_MEMBERS="${EXTRA_VAR_MEMBERS:-mean}"  # mean | all  (only for save=all)
N_MEMBERS="${N_MEMBERS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-data/predictions}"

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

LOGDIR="logs/2024"
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

echo "$(timestamp)  Starting 2024 benchmark  [${START} → ${END}]  (parallel)"
echo "$(timestamp)  Models: ${MODELS[*]}"
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
