# Launch NeuralGCM inference

set -euo pipefail

cd "$(dirname "$0")"                  

# ── Config ────────────────────────────────────────────────────────────────────
ENV="${ENV:-/home/ubuntu/miniconda3/envs/neuralgcm}"   # dedicated NeuralGCM env
GPU="${GPU:-1}"                          
START="${START:-2024-01-01}"
END="${END:-2024-12-24}"
LEAD_DAYS="${LEAD_DAYS:-1 3 5 7}"
N_MEMBERS="${N_MEMBERS:-10}"               # match GenCast
SAVE_VARIABLES="${SAVE_VARIABLES:-all}"    # save every variable, regridded to EA grid
EXTRA_VAR_MEMBERS="${EXTRA_VAR_MEMBERS:-mean}"  # non-precip = ensemble mean; precip = all members
OUTPUT_DIR="${OUTPUT_DIR:-data/predictions}"    # writes to <dir>/neuralgcm/

PYTHON="${ENV}/bin/python"
SITE="${ENV}/lib/python3.11/site-packages"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: NeuralGCM env not found at ${ENV}" >&2
    echo "Create it first:" >&2
    echo "  conda create -n neuralgcm python=3.11 -y" >&2
    echo "  conda run -n neuralgcm pip install neuralgcm \"jax[cuda12]\" gcsfs xarray zarr xesmf" >&2
    exit 1
fi
if [[ "${GPU}" == "0" ]]; then
    echo "ERROR: GPU 0 is used by GenCast. Pick a free GPU (e.g. GPU=1)." >&2
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
# Use ONLY this env's packages (block ~/.local from shadowing its JAX/cuDNN).
export PYTHONNOUSERSITE=1
# Point the loader at THIS env's bundled NVIDIA libs (matching jax[cuda12] 0.6.2),
# so cuDNN resolves correctly (the earlier "Unable to load cuDNN" failure was a
# cuDNN/JAX version mismatch from the shared env).
export LD_LIBRARY_PATH=\
${SITE}/nvidia/cudnn/lib:\
${SITE}/nvidia/cublas/lib:\
${SITE}/nvidia/cuda_runtime/lib:\
${SITE}/nvidia/cuda_nvrtc/lib:\
${SITE}/nvidia/cufft/lib:\
${SITE}/nvidia/cusolver/lib:\
${SITE}/nvidia/cusparse/lib:\
${SITE}/nvidia/nccl/lib:\
${SITE}/nvidia/nvjitlink/lib:\
${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONWARNINGS="ignore::FutureWarning"

mkdir -p logs/2024

echo "$(date '+%F %T')  NeuralGCM inference"
echo "  env       : ${ENV}"
echo "  GPU       : ${GPU}   (GenCast is on GPU 0)"
echo "  dates     : ${START} → ${END}"
echo "  members   : ${N_MEMBERS}"
echo "  save vars : ${SAVE_VARIABLES} (extra-var members: ${EXTRA_VAR_MEMBERS})"
echo "  output    : ${OUTPUT_DIR}/neuralgcm/"
echo ""

exec "${PYTHON}" -m benchmark_ea.run \
    --models neuralgcm \
    --start  "${START}" \
    --end    "${END}" \
    --lead-days ${LEAD_DAYS} \
    --n-members "${N_MEMBERS}" \
    --save-variables "${SAVE_VARIABLES}" \
    --extra-var-members "${EXTRA_VAR_MEMBERS}" \
    --output-dir "${OUTPUT_DIR}"
