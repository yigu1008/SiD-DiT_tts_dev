# Package Installation Scripts

## Dedicated installers

| Script | Purpose | When to use |
|--------|---------|-------------|
| `install_reward_deps.sh` | Installs reward backends: ImageReward, CLIP, hpsv2, timm, wandb, qwen-vl-utils, openai. Has stamp+quick-verify to skip on re-runs (`~/.cache/sid_deps/reward_deps_ok_v2`). Uses `constraints.txt` to prevent silent downgrades of ftfy/regex/xxhash. | Run by every AMLT yaml via `PYTHON_BIN=... bash install_reward_deps.sh`. Force re-run: `FORCE_INSTALL_DEPS=1`. |
| `prepare_cluster_overlay_deps.sh` | Installs a subset of fragile deps (xxhash, ftfy, regex, pandas, pyarrow, datasets, httpx, wandb, timm) into a user-writable `--target` dir when the conda env is read-only. | Fallback path in `install_reward_deps.sh` when `wandb` reinstall fails due to permissions. Also triggered by `ENABLE_OVERLAY_PYDEPS=1` in yamls. Set `SID_EXTRA_PYTHONPATH` to the output dir before launching. |

## Environment setup / repair (local / TACC)

| Script | Purpose |
|--------|---------|
| `rebuild_sid_env.sh` | Full nuclear reset: removes and recreates the `sid_dit` conda env from scratch. Use when the env is corrupted. |
| `fix_cudnn_stack.sh` | Force-reinstalls torch + cuDNN/CUDA wheels to fix mismatched CUDA stacks without rebuilding the whole env. |
| `tacc_setup.sh` | TACC/SSH cluster env setup (mirrors the AMLT yaml preamble). `source tacc_setup.sh` to set env vars and run installs. `SKIP_INSTALL=1 source tacc_setup.sh` for env vars only. |

## Incidental installs (side effects inside suite scripts)

These are not installers per se but contain inline `ensure_*_runtime` functions with pip calls:

| Script | What it installs |
|--------|-----------------|
| `hpsv2_sd35_sid_ddp_suite.sh` | `hpsv2` (ensure_hpsv2_runtime), `xformers` (ensure_xformers), `regex>=2024.11.6` (ensure_qwen_precompute_runtime) |
| `hpsv2_flux_schnell_ddp_suite.sh` | same pattern as above |
| `hpsv2_sana_sid_ddp_suite.sh` | same pattern as above |
| `get_hpsv2_prompts.sh` | `hpsv2` (no-deps first, then with deps as fallback) |

## Key files referenced by installers

| File | Role |
|------|------|
| `requirements.txt` | Main pip requirements; pinned at job start via `pip install -r requirements.txt` |
| `constraints.txt` | Lower-bound floor for ftfy/regex/xxhash; enforced by `_pip()` wrapper in `install_reward_deps.sh` |
