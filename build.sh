#!/usr/bin/env bash
# Builds the image used by compose.yaml: vLLM's OpenAI-compatible server with
# Inkling 2 Mini support, compiled for SM120 (RTX PRO 6000 Blackwell).
#
# Inkling 2 Mini support is in vllm-project/vllm#69028, which is not in a vLLM
# release yet, and no published x86_64 vLLM image has both that support and
# SM120 kernels. This builds the PR from source at a tested commit, with the
# fixes in patches/ applied.
#
# Optional environment variables:
#   VLLM_REF      vLLM commit (full SHA) or ref to build; default: tested PR commit
#   MAX_JOBS      parallel compile jobs; default: one per CPU core, but no more
#                 than one per 8 GB of RAM
#   NVCC_THREADS  threads per CUDA compile job; default: 2
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/vllm"
VLLM_REF="${VLLM_REF:-b426b46e63408c41b0aec069fdaf24dd640abe58}"
IMAGE=vllm-inkling2-mini:sm120  # the tag compose.yaml runs
ARCHS="12.0"                    # SM120

mem_jobs=$(awk '/^MemTotal:/ {print int($2 / 1024 / 1024 / 8)}' /proc/meminfo)
jobs=$(( mem_jobs < $(nproc) ? mem_jobs : $(nproc) ))
MAX_JOBS="${MAX_JOBS:-$(( jobs > 0 ? jobs : 1 ))}"
NVCC_THREADS="${NVCC_THREADS:-2}"

DOCKER=docker
if ! docker info >/dev/null 2>&1; then
  echo "Can't use Docker as $(id -un); running docker with sudo."
  DOCKER="sudo docker"
fi

if [[ ! -d "${SRC}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/vllm-project/vllm.git "${SRC}"
fi
# Fetch the exact commit, so the build keeps working after the PR branch moves.
git -C "${SRC}" fetch -q origin "${VLLM_REF}"
git -C "${SRC}" checkout -q -f FETCH_HEAD
for p in "${HERE}"/patches/*.patch; do
  [[ -e "$p" ]] || continue
  git -C "${SRC}" apply --whitespace=nowarn "$p"
  echo "Applied $(basename "$p")"
done
COMMIT="$(git -C "${SRC}" rev-parse HEAD)"
echo "Building ${IMAGE} from vLLM ${COMMIT} + patches/ (MAX_JOBS=${MAX_JOBS})"

DOCKER_BUILDKIT=1 ${DOCKER} build "${SRC}" \
  -f "${SRC}/docker/Dockerfile" \
  --target vllm-openai \
  --build-arg torch_cuda_arch_list="${ARCHS}" \
  --build-arg max_jobs="${MAX_JOBS}" \
  --build-arg nvcc_threads="${NVCC_THREADS}" \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg VLLM_BUILD_COMMIT="${COMMIT}" \
  --build-arg VLLM_IMAGE_TAG="${IMAGE}" \
  --progress=plain \
  -t "${IMAGE}"

echo "Built ${IMAGE}"
