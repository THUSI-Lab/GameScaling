# GameScaling Evaluation Suite

Evaluation code for four closed-loop game benchmarks:

| Directory | Benchmark | Game environment |
|---|---|---|
| `openha_minecraft/` | OpenHA (Minecraft) | MineStudio 1.1.6 (Java Minecraft) |
| `bench_eval/vizdoom/` | ViZDoom | vizdoom 1.2.4 |
| `bench_eval/omnigamearena/` | OmniGameArena (OGA) | Unreal Engine 5 packaged games |
| `bench_eval/gameworld/` | GameWorld | 34 browser games (headless Chromium) |

In every benchmark the model is served by vLLM as an OpenAI-compatible endpoint.
The harness captures game frames, sends them to the model, parses the returned
action string (`<|action_start|> ... <|action_end|>`), and replays the action in
the game. This repository contains code, configs and small assets only. It does
not contain model weights, results or logs.

## Release scope and evaluation protocols

This release contains the GameScaling evaluation adapters, benchmark task
configurations, and launch scripts. The default `scripts/run_bench.sh` workflow
uses **time-paused evaluation**: `pdq` for ViZDoom and OmniGameArena, `paused`
for GameWorld, and `stop` pacing for OpenHA. Model inference time is not charged
to the game clock in these default runs.

- **ViZDoom real time:** the retained `rt` option is the legacy, synchronous
  inference-time-compensated implementation. The environment pauses for the
  model call and advances afterward to account for the measured inference
  latency. This release does **not** include the updated continuously running
  ViZDoom real-time implementation used for the corrected real-time results
  in the paper. The legacy `rt` option must not be presented as reproducing
  those corrected results.
- **GameWorld task set:** the included configuration follows the earlier
  **34 games, 170 tasks, 100-step-per-task** protocol. It is not the current
  upstream 400-task benchmark. Task coverage and step budgets must be stated
  when reporting or comparing scores.
- **External resources:** model checkpoints, GameWorld browser-game assets,
  the OmniGameArena UE5 build, and the Minecraft runtime are obtained
  separately as described below. This source release is not a self-contained
  distribution of those models or game environments.

The commands below reproduce the protocols included in this release. They do
not, by themselves, reproduce every protocol variant or result in the paper.

## Repository layout

```
.
├── README.md                 this file: setup and reproduction procedure
├── REQUIREMENTS.md           per-benchmark system dependencies and notes
├── requirements/             pip requirement files, one per benchmark
├── scripts/
│   ├── setup_workspace.sh    builds the bench workspace expected by bench_eval drivers
│   ├── run_bench.sh          runs one benchmark with the fixed protocol
│   └── patch_minestudio.py   NumPy 2.x fix for MineStudio 1.1.6
├── openha_minecraft/         OpenHA client (openagents package), serving and eval scripts
└── bench_eval/
    ├── vizdoom/              harness, WAD maps, driver (docker/eval_vzd.sh)
    ├── omnigamearena/        harness, configs, UE5 helpers, driver (docker/eval_parallel.sh)
    └── gameworld/            harness, task catalog, driver (docker/eval_ckpt.sh)
```

## External resources (not included)

| Resource | Needed by | Source |
|---|---|---|
| Model checkpoint | all | Merged Hugging Face checkpoint. Its tokenizer must contain `<|action_start|>` and `<|action_end|>`. |
| UE5 game build (UE 5.5.4, Linux) | OGA | https://huggingface.co/datasets/mxlin043/OmniGameArena or https://www.modelscope.ai/datasets/mxlin043/OmniGameArena |
| GameWorld games (~166 MB, 34 games) | GameWorld | https://github.com/gameworld-project/GameWorld-Games |
| MCP-Reborn engine | OpenHA | Built automatically by MineStudio on the first environment reset |
| Replay-video font `SimHei.ttf` | OpenHA, only when rendering replay videos | [CraftJarvis/OpenHA](https://github.com/CraftJarvis/OpenHA) (`openagents/assets/fonts/SimHei.ttf`); see below |

## Hardware

- One node with 8 NVIDIA GPUs (the reference runs used H200 / H20-class GPUs), at least 96 CPU cores, at least 512 GB RAM, and at least 100 GB free on local `/tmp`.
- OGA additionally needs a host driver with graphics (Vulkan) support. See [REQUIREMENTS.md](REQUIREMENTS.md#3-omnigamearena--bench_evalomnigamearena).

Each benchmark uses all 8 GPUs, so run the benchmarks one after another, not in parallel.

## Reproduction

The commands below assume the repository is at `$EVAL` and the bench workspace is at `/tmp/bench_ws`.

### 1. Model-serving environment

Install vLLM with a matching torch / transformers. The reference runs used vLLM 0.17.0 with torch 2.10.0+cu129 (for example the `verlai/verl:vllm017.latest` image). Qwen3.5-based checkpoints need vLLM 0.12 or newer, and 0.8.5 cannot load them.

```bash
python3 -c "import vllm; print(vllm.__version__)"
```

### 2. Benchmark dependencies

```bash
export EVAL=/path/to/this/repo

# ViZDoom, OGA, GameWorld harnesses (these do not modify vLLM / torch)
pip install -r $EVAL/requirements/vizdoom.txt
pip install -r $EVAL/requirements/omnigamearena.txt
pip install -r $EVAL/requirements/gameworld.txt
python -m playwright install --with-deps chromium

# OpenHA: keep this order
apt-get update && apt-get install -y openjdk-8-jdk-headless xvfb
pip install --no-deps minestudio==1.1.6
pip install -r $EVAL/requirements/openha_minecraft.txt   # pins numpy==1.26.4 last
(cd $EVAL/openha_minecraft && pip install --no-deps -e .)
python3 $EVAL/scripts/patch_minestudio.py
```

`requirements/openha_minecraft.txt` pins NumPy 1.26.4. If the vLLM installation requires NumPy 2.x, install the OpenHA client in a separate virtual environment and keep vLLM in the system environment.

**OpenHA replay videos (optional).** The evaluation protocol runs with `--no_render` (`NO_RENDER=1` in `scripts/run_bench.sh`), so it does not need any font. To render replay videos with text overlays, first download the font from the upstream OpenHA repository:

```bash
mkdir -p $EVAL/openha_minecraft/openagents/assets/fonts
wget -O $EVAL/openha_minecraft/openagents/assets/fonts/SimHei.ttf \
  https://raw.githubusercontent.com/CraftJarvis/OpenHA/main/openagents/assets/fonts/SimHei.ttf
```

SimHei is a commercial typeface, so it is not redistributed here. Check its license before use.

OGA also needs Vulkan, GL/X and ffmpeg system libraries. See [REQUIREMENTS.md](REQUIREMENTS.md#3-omnigamearena--bench_evalomnigamearena).

### 3. External game assets

```bash
# GameWorld games. The harness reads bench_eval/gameworld/games/benchmark/
cd $EVAL/bench_eval/gameworld
git clone https://github.com/gameworld-project/GameWorld-Games.git games/gameworld-games
ln -sfn gameworld-games/benchmark games/benchmark
ls games/benchmark/01_2048/index.html          # 34 game directories expected

# OGA UE5 build (only needed for OGA); use one of the two
huggingface-cli download --repo-type dataset mxlin043/OmniGameArena --local-dir /tmp/omnigamearena_ue5
modelscope download --dataset mxlin043/OmniGameArena --local_dir /tmp/omnigamearena_ue5
```

After the download, `/tmp/omnigamearena_ue5/Linux/OmniGameArena.sh` must exist and be executable. Extract the archive and run `chmod +x` on it if needed.

### 4. Bench workspace

The bench_eval drivers expect the code under `$WS/evaluation/<bench>` together with a `.bench-workspace-id` marker file. The following script creates that layout and links the UE5 build:

```bash
bash $EVAL/scripts/setup_workspace.sh /tmp/bench_ws /tmp/omnigamearena_ue5
```

Re-run it after changing code under `bench_eval/` or after adding the game assets, because it copies the code into the workspace.

### 5. Run

```bash
export MODEL_DIR=/path/to/merged_hf_checkpoint TAG=my_ckpt

# Smoke test: a few minutes per benchmark. It checks that the pipeline works; scores are not meaningful.
for b in vizdoom gameworld openha oga; do SMOKE=1 bash $EVAL/scripts/run_bench.sh $b; done

# Full runs (a terminal multiplexer such as tmux is recommended)
bash $EVAL/scripts/run_bench.sh vizdoom     # ~30-40 min
bash $EVAL/scripts/run_bench.sh oga         # ~1-2 h
bash $EVAL/scripts/run_bench.sh gameworld   # ~1-2 h
bash $EVAL/scripts/run_bench.sh openha      # several hours
```

`SMOKE=1` writes to the same result directories as a full run. Move or delete the smoke results before starting the full run, otherwise the aggregators include the smoke episodes.

The GPU layout defaults assume 8 GPUs. On smaller nodes, override the replica counts: `VZD_REPLICAS` (default 4), `GW_PORTS` (default `8088 ... 8095`, one vLLM per port; with a single port also set `SFT_ENDPOINT=http://127.0.0.1:8088/v1/chat/completions` so every shard uses it), and `OPENHA_DP` (default 8). OGA needs at least 2 GPUs (one for vLLM, at least one for UE5). Results from a non-default layout should be reported as such.

After each benchmark, confirm with `nvidia-smi` that all vLLM processes have exited. The OpenHA server is not stopped automatically; stop it with `pkill -f 'vllm serve'`.

### Evaluation protocol

These settings are fixed in `scripts/run_bench.sh`. "a5-480p" means 854x480 model input with a 5-frame image history.

| Benchmark | Scale | Settings | GPU layout |
|---|---|---|---|
| ViZDoom | 4 maps x 10 episodes, 350 steps per episode | a5-480p, `pdq` clock, free-form thinking, temperature 1, projective mouse | 4 vLLM replicas; Doom runs on CPU |
| OGA | 7 solo games x 10 episodes | a5-480p (resize 854x480), `pdq` clock, free-form thinking, temperature 1 | 3 vLLM replicas + 5 UE5 instances, one UE5 instance per GPU |
| GameWorld | 34 games x 5 tasks = 170 tasks, 100 steps per task | 480p, paused clock, thinking enabled, temperature 1 | 8 vLLM replicas; served model name `/model` |
| OpenHA | 30 tasks (`task_list.json`) x 5 rollouts, 600 steps per rollout | 480p model input, optional thinking, history of 3 (actions only), calibrated mouse mapping, temperature 0.9 | 1 vLLM server, TP=1, DP=8 |

OpenHA is reported at 480p by default. Set `OPENHA_RES=720p` to run the 720p variant (1280x720 model input). The Minecraft environment itself always renders at 640x360; resizing happens only at the model input.

### Results and acceptance checks

| Benchmark | Result file | Primary metric | Completeness check |
|---|---|---|---|
| ViZDoom | `$WS/evaluation/vizdoom/eval_summary_${TAG}_doom_a5_480p.json` | Per-map mean (kill count for battle maps, reward for defend maps); `overall` is for reference only | `python3 docker/vzd_check_sanity.py runs/doom <served_name> --episodes 10` prints `VERDICT: OK`; 40 episodes in total |
| OGA | `$WS/evaluation/omnigamearena/eval_summary_${TAG}_oga_a5_480p_e10.json` | `overall_mean` and per-game means | 70 episodes; `python3 docker/oga_check_render.py omnigamearena/runs/yourvla <served_name>` reports no render failures or stalled episodes |
| GameWorld | `$WS/evaluation/gameworld/results/${TAG}_gw_a5_480p/${TAG}/eval_summary.json` | SR and PG (reported score = PG x 100) | `completion_gate.json` passes: 170 tasks, none errored or missing |
| OpenHA | `/tmp/openha_eval_out/${TAG}_480p_30tasks/eval_summary_*.json` | `overall` success rate and per-category rates | `overall.total` = 150 |

Results under `/tmp` are not persistent; copy them elsewhere after the run.

## Troubleshooting

- **`[BENCH][FATAL] own code workspace not mounted`**: the workspace was not created, or `WS_OVERRIDE` is not set. `run_bench.sh` sets it automatically; when calling a driver directly, run `export WS_OVERRIDE=/tmp/bench_ws BENCH_CODE_HINT=/tmp/bench_ws`.
- **OGA scores drop by about half without errors**: two UE5 instances share a GPU. Keep `PAR + VLLM_REPLICAS <= 8`.
- **`vulkaninfo` lists only llvmpipe**: the host driver lacks graphics support, or the container was started without `NVIDIA_DRIVER_CAPABILITIES=all`. See REQUIREMENTS.md §3.
- **OpenHA `sqlite3.OperationalError: disk I/O error`**: `MINESTUDIO_DIR` is on a network or FUSE file system. It must be on a local disk such as `/tmp`.
- **OpenHA node becomes unresponsive at high concurrency**: `LP_NUM_THREADS=2` is not in effect.
- **First OpenHA reset is slow**: MineStudio is building MCP-Reborn. This happens once per `MINESTUDIO_DIR`.

## Implementation notes

Issues found while validating all four benchmarks on vLLM 0.17.0 / torch 2.10.0+cu129 with 8 GPUs, and how this code handles them:

1. **vLLM port collision when replicas start together.** Starting several `vllm serve` replicas at the same moment can make the vLLM V1 `EngineCore` pick colliding internal `torch.distributed` ports (`TCPStoreLibUvBackend.cpp: EADDRINUSE`, exit code 27). The ViZDoom and OGA drivers stagger replica start-up with `sleep "${VLLM_LAUNCH_STAGGER:-10}"`.
2. **OGA resolution and step limits.** `oga_make_configs.py` expects `RESIZE` as `WxH`, so `run_bench.sh` passes `RESIZE=854x480`. `eval_parallel.sh` forwards `MAX_STEPS` to `run_benchmark.py`, which falls back to the per-game `GameSpec.max_steps` only when `--max-steps` is not given.
3. **GameWorld runtime installation.** `eval_ckpt.sh` skips the `pip install` / `playwright install` step when `colorlog`, `playwright` and the Chromium binary are already present, so that pre-installed package versions are not changed. Smoke runs use the 2-task suite `qwen3-vl-8b-sft8-gsmoke` (`01_2048`, `07_chrome-dino`).
4. **Containers without NVIDIA Vulkan user-space libraries.** Containers started with `NVIDIA_DRIVER_CAPABILITIES=compute,utility` lack `libGLX_nvidia.so.0`, `nvidia_icd.json` and `/dev/dri/*`, and UE5 then exits because it finds no Vulkan GPU. Restarting the container with `NVIDIA_DRIVER_CAPABILITIES=all` fixes this. Alternatively, without a restart: extract `libnvidia-gl-<driver_version>` that exactly matches the host driver into `/usr`, write `/etc/vulkan/icd.d/nvidia_icd.json`, and create `/dev/dri/card*` and `/dev/dri/renderD*` with `mknod`.
5. **OpenHA installation.** Without network access for build isolation, install `openagents` with `pip install --no-build-isolation --no-deps -e .`. Smoke runs use 40 steps per rollout (full runs use 600).

## License and third-party components

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and retained license texts. Third-party code and assets retain their own terms; the repository-root license does not replace them.

`openha_minecraft/` is derived from [CraftJarvis/OpenHA](https://github.com/CraftJarvis/OpenHA) (`main` branch, MIT License), with modifications for the evaluation protocol described here; it builds on MineStudio. The GameWorld games, the OmniGameArena UE5 build and ViZDoom / Freedoom are third-party works distributed under their own licenses and are obtained separately as described above.
