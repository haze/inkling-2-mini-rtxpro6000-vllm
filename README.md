# Inkling 2 Mini on an RTX PRO 6000 (vLLM + Docker)

Serves [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
through vLLM's OpenAI-compatible API on one NVIDIA RTX PRO 6000 Blackwell, with
the model's full 1M-token context, up to 999 images per prompt and DSpark
speculative decoding.

Inkling 2 Mini support
([#69028](https://github.com/vllm-project/vllm/pull/69028)) is in vLLM main but
not in a release yet, and no published x86_64 vLLM image has both that support
and kernels for this GPU (SM120). `build.sh` builds vLLM main from source at a
pinned commit for SM120, with one fix from `patches/` (it stops the MoE kernels
from recompiling for every new prompt length). The pinned commit includes the
SM120 speedups from [#69021](https://github.com/vllm-project/vllm/pull/69021)
and [#69168](https://github.com/vllm-project/vllm/pull/69168). `compose.yaml`
runs the server.

## Prerequisites

- **GPU:** NVIDIA RTX PRO 6000 Blackwell (96 GB), otherwise idle. The server
  takes 92% of its memory.
- **NVIDIA driver** with CUDA 13.2 or newer.
- **Docker Engine** with the Buildx and Compose v2 plugins, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  configured for Docker. Your user should be able to run `docker` without sudo
  (`sudo usermod -aG docker $USER`, then log out and back in).
- **git** and the Hugging Face CLI `hf`: `curl -LsSf https://hf.co/cli/install.sh | bash`
  or `pip install -U huggingface_hub`.
- **Disk:** ~200 GB free: ~20 GB of weights, a ~37 GB image and up to ~120 GB
  of Docker build cache (you can delete the cache after the build).
- **Time:** the build compiles vLLM's CUDA kernels, ~45 minutes on a 20-core
  machine. It runs one compile job per CPU core, at most one per 8 GB of RAM.

To check them:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv  # the GPU, almost no memory used
nvidia-smi | grep -o 'CUDA Version: [0-9.]*'                       # 13.2 or higher
docker buildx version && docker compose version                    # both plugins installed
docker run --rm --gpus all ubuntu nvidia-smi -L                    # containers can see the GPU
hf version
df -h .                                                            # ~200 GB available
```

Tested on x86_64 Linux with driver 595.91.07, Docker 29.1, Docker Compose 2.40
and NVIDIA Container Toolkit 1.20.

## Set up

In a clone of this repo:

```bash
# 1. Download the model (~20 GB) at the pinned revision.
hf download thinkingmachines/Inkling-2-Mini \
    --revision b85c3cfa615fc413cc9af3379f491b963f3b586a \
    --local-dir models/Inkling-2-Mini

# 2. Build the image, vllm-inkling2-mini:sm120 (~45 minutes).
./build.sh
```

A successful build ends with `Built vllm-inkling2-mini:sm120`. If it fails with
`Killed` or `signal 9`, the machine ran out of memory: use fewer compile jobs,
e.g. `MAX_JOBS=4 ./build.sh`.

To build the newest vLLM main instead of the pinned commit, run
`VLLM_REF=main ./build.sh`. If the patch in `patches/` no longer applies, the
build stops before compiling.

## Run

```bash
docker compose up --wait   # returns once the server is healthy
```

The first start takes ~5 minutes because kernels compile (`docker compose logs -f`
shows progress). They're cached in Docker volumes, so later starts take 1-2
minutes. If `up` reports an error, `docker compose logs` shows the cause. Check
that the server answers:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "thinkingmachines/Inkling-2-Mini",
       "messages": [{"role": "user", "content": "Say hello."}],
       "reasoning_effort": "low"}'
```

- **Endpoint:** point OpenAI-compatible clients at `http://<host>:8000/v1`,
  model `thinkingmachines/Inkling-2-Mini`.
- **Reasoning:** the model's reasoning comes back in `message.reasoning`, and
  `reasoning_effort` sets how much it thinks.
- **Long prompts:** a prompt close to 1M tokens takes ~5-6 minutes before the
  first output token, so set client timeouts to 10+ minutes.

There is no API key, and the server listens on all interfaces. To change that
or other settings, put them in a `.env` file next to `compose.yaml`:

| variable | default | meaning |
|---|---|---|
| `INKLING_MODEL_DIR` | `./models/Inkling-2-Mini` | model directory, if you downloaded it elsewhere |
| `INKLING_PORT` | `8000` | host port |
| `INKLING_BIND` | `0.0.0.0` | host address to listen on; `127.0.0.1` allows local access only |
| `INKLING_GPU` | `0` | GPU index |

## Tear down

```bash
docker compose down   # stop the server and free the GPU
```

To also remove everything this setup created:

```bash
docker compose down -v               # also deletes the kernel-cache volumes
docker rmi vllm-inkling2-mini:sm120
docker builder prune -a              # build cache; this clears ALL unused Docker build cache
rm -rf vllm models                   # vLLM source checkout and model weights
```
