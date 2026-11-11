# Environment Requirements

All four benchmarks are closed-loop interactive evaluations. Each one has two parts:

1. **Model server**: vLLM + torch + transformers on GPU, shared by all four benchmarks.
2. **Game environment + harness**: different for each benchmark, described below.

Pip requirement files are in `requirements/<bench>.txt`. For the end-to-end procedure, see [README.md](README.md).

## Overview

| Benchmark | Game environment | Rendering | System dependencies | External resources (not included) | GPU layout |
|---|---|---|---|---|---|
| OpenHA (Minecraft) | MineStudio 1.1.6 (Java Minecraft) | CPU (xvfb + llvmpipe) | JDK 8, xvfb | MCP-Reborn engine (built on first reset) | GPUs run vLLM only |
| ViZDoom | vizdoom 1.2.4 | CPU software rendering | none | none (WAD maps included) | GPUs run vLLM only, 4 replicas by default |
| OmniGameArena | UE5 packaged games | **GPU, Vulkan** | Vulkan/GL/X user-space libraries, xvfb, ffmpeg; host NVIDIA driver with graphics support | **UE5 game build** (Hugging Face / ModelScope `mxlin043/OmniGameArena`) | 8 GPUs = 3 vLLM + 5 UE5, **one UE5 instance per GPU** |
| GameWorld | 34 browser games | CPU (headless Chromium) | Playwright Chromium | **Game files, ~166 MB** (`gameworld-project/GameWorld-Games`) | 8 GPUs = 8 vLLM replicas |

## 0. Model server (shared)

- CUDA 12.x GPUs (reference runs: H200 / H20-class).
- `vllm` with a matching `torch` / `transformers`. Versions known to work:
  - **vLLM 0.17 / torch 2.10** (recommended; e.g. image `verlai/verl:vllm017.latest`);
  - vLLM 0.12.
  - vLLM 0.8.5 cannot load Qwen3.5 and is not supported.
- Serving reference: `openha_minecraft/serve_qwen3vl.sh`, and the `vllm serve` sections of `bench_eval/*/docker/eval_*.sh`.

## 1. OpenHA (Minecraft): `openha_minecraft/`

```bash
apt-get update && apt-get install -y openjdk-8-jdk-headless xvfb
pip install --no-deps minestudio==1.1.6           # --no-deps is required; its av/opencv pins conflict with vLLM
pip install -r requirements/openha_minecraft.txt  # numpy==1.26.4 is pinned last
cd openha_minecraft && pip install --no-deps -e . # installs the openagents package
python3 ../scripts/patch_minestudio.py            # required, see below
```

**MineStudio must be patched.** Without the patch it fails partway through a rollout under NumPy 2.x. In `minestudio/simulator/minerl/herobraine/hero/spaces.py`, `Text.sample` calls `np.array(..., np.dtype)`; this must be `np.array(..., dtype=self.dtype)`. `scripts/patch_minestudio.py` applies the change and is idempotent.

Runtime environment variables:

```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export MINESTUDIO_DIR=/tmp/minestudio   # must be on a local disk; SQLite locking fails on FUSE / network file systems
export LP_NUM_THREADS=2                 # llvmpipe threads; required at high concurrency
export MINESTUDIO_MAX_WAIT=120
```

Notes:
- The first environment reset builds MCP-Reborn under `MINESTUDIO_DIR`. This is slow, but it happens only once per directory.
- SAM2 (`--sam_path`) is an optional dependency and is not used by the evaluation.
- Replay-video rendering needs `openagents/assets/fonts/SimHei.ttf`, which is not shipped. Download it from [CraftJarvis/OpenHA](https://github.com/CraftJarvis/OpenHA/tree/main/openagents/assets/fonts) (see README.md). The evaluation itself runs with `--no_render` and does not need it.
- The task list is `CrossAgent/STRL/data_processor/utils/task_list.json` (30 tasks).

## 2. ViZDoom: `bench_eval/vizdoom/`

```bash
pip install -r requirements/vizdoom.txt
```

- Pure CPU software rendering. No Vulkan, `/dev/dri`, X server or apt packages are needed. One Doom instance uses about 40 MB of RAM.
- The maps `vizdoom_arena/scenarios/*.wad` (D3_battle, D4_battle2, defend_the_center, defend_the_line) are included. The base IWAD `freedoom2.wad` and the engine ship inside the vizdoom wheel.
- For offline machines, provide a `vizdoom==1.2.4` wheel yourself.

## 3. OmniGameArena: `bench_eval/omnigamearena/`

```bash
pip install -r requirements/omnigamearena.txt
apt-get install -y libvulkan1 vulkan-tools mesa-vulkan-drivers xvfb ffmpeg   # plus GL/X runtime libraries
```

- **The UE5 game build must be downloaded separately.** It is the publicly released UE 5.5.4 Linux build, available from either source:
  - Hugging Face: https://huggingface.co/datasets/mxlin043/OmniGameArena
  - ModelScope: https://www.modelscope.ai/datasets/mxlin043/OmniGameArena

  ```bash
  # either one
  huggingface-cli download --repo-type dataset mxlin043/OmniGameArena --local-dir omnigamearena_ue5
  modelscope download --dataset mxlin043/OmniGameArena --local_dir omnigamearena_ue5
  ```

  After the download, check that `omnigamearena_ue5/Linux/OmniGameArena.sh` exists and is executable (extract it or `chmod +x` it if needed). Then either pass the directory to `scripts/setup_workspace.sh`, or set `export BENCH_UE5_DIR=/path/to/omnigamearena_ue5/Linux`. By default the driver looks in `$WS/evaluation/omnigamearena/omnigamearena_ue5/Linux`.
- **Vulkan is required:**
  - The container needs `NVIDIA_DRIVER_CAPABILITIES=all`. With `compute` only, UE5 does not start.
  - The host driver must include graphics support: both `/usr/share/vulkan/icd.d/nvidia_icd.json` and `libGLX_nvidia.so.0` must exist.
  - If `vulkaninfo` lists only llvmpipe, the driver setup is wrong.
- **Run one UE5 instance per GPU.** If two instances share a GPU, scores drop by about half and no error is reported. Standard layout: 8 GPUs = 3 vLLM replicas + 5 UE5 instances.
- Running several instances needs the port shim `docker/ue5_port_shim.c` / `.so` (included). It is loaded with `LD_PRELOAD` and moves the hard-coded RemoteInput port 12345 to a per-instance port.
- A suitable image can be built `FROM verlai/verl:vllm017.latest` by adding the Vulkan/GL/X libraries, xvfb and ffmpeg.

## 4. GameWorld: `bench_eval/gameworld/`

```bash
# Python >= 3.12
pip install -r requirements/gameworld.txt
python -m playwright install --with-deps chromium
```

- 34 browser games and 170 tasks. The task configuration (`catalog/`) is included. **The game files (~166 MB) are not included** and must be placed under `bench_eval/gameworld/games/benchmark/`:
  ```bash
  cd bench_eval/gameworld
  git clone https://github.com/gameworld-project/GameWorld-Games.git games/gameworld-games
  ln -sfn gameworld-games/benchmark games/benchmark   # the harness reads games/benchmark/
  ls games/benchmark/01_2048/index.html                # 34 game directories expected
  ```
- The harness drives headless Chromium and uses CPU only.
- The vLLM `served-model-name` is fixed to `/model`.
- SDKs are needed only to evaluate closed API models: `openai`, `anthropic`, `google-genai`, `zai-sdk`.

## 5. Workspace check

Each `bench_eval/*/docker/eval_*.sh` driver starts with a `BENCH_WORKSPACE` check. The check requires `$WS/.bench-workspace-id` and the code under `$WS/evaluation/<bench>`. Create the workspace with `scripts/setup_workspace.sh`, then point the drivers at it with `WS_OVERRIDE`.
