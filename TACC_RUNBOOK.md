# TACC Runbook

## Conda quick start (TACC login node)

Use this when `conda` is not yet available in your shell:

```bash
# 1) Find conda.sh
for p in \
  "$SCRATCH/miniconda3/etc/profile.d/conda.sh" \
  "$SCRATCH/anaconda3/etc/profile.d/conda.sh" \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  [[ -f "$p" ]] && echo "$p"
done

# 2) Source the first one that exists
CONDA_SH="$(
  for p in \
    "$SCRATCH/miniconda3/etc/profile.d/conda.sh" \
    "$SCRATCH/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    [[ -f "$p" ]] && { echo "$p"; break; }
  done
)"
if [[ -z "$CONDA_SH" ]]; then
  echo "No conda.sh found. Install Miniconda or set CONDA_SH manually."
  exit 1
fi
source "$CONDA_SH"

# 3) Activate env
conda activate sid_dit

# 4) Verify
which python
python -V
conda env list
```

---

## One-time setup (login node, done once per env rebuild)

```bash
# 1. Activate conda env
conda activate sid_dit

# 2. Source setup to set DATA_ROOT, HF_HOME, IMAGEREWARD_CACHE, etc.
source ~/SiD-DiT_tts_dev/tacc_setup.sh   # SKIP_INSTALL=0 to run pip installs

# 3. Pre-download all reward models to $WORK (needs internet — login node only)
python ~/SiD-DiT_tts_dev/preload_reward_models.py

# 4. Pre-download Qwen rewrite model to $WORK (needed for prompt-variant ablations)
huggingface-cli download Qwen/Qwen3-4B
```

Model cache locations (all on `$WORK`, not home):
| Model | Path |
|---|---|
| HuggingFace (SD3.5, Flux, Qwen) | `$WORK/ls6/model_cache/huggingface/` |
| ImageReward | `$WORK/ls6/model_cache/ImageReward/ImageReward.pt` |
| PickScore | `$WORK/ls6/model_cache/huggingface/hub/models--yuvalkirstain--PickScore_v1/` |
| HPSv2 | `$WORK/ls6/model_cache/hpsv2/` |
| CLIP | `$WORK/ls6/model_cache/clip/` → symlinked to `~/.cache/clip` |
| ImageReward | also symlinked to `~/.cache/ImageReward` |

---

## Running jobs

### Full HPSv2 run (SD3.5 + Flux, all algorithms)

```bash
bash ~/SiD-DiT_tts_dev/run_tacc.sh
```

Key overrides:
```bash
METHODS="baseline mcts" \
STEPS=4 N_VARIANTS=3 N_SIMS=50 \
START_INDEX=0 END_INDEX=50 \
bash ~/SiD-DiT_tts_dev/run_tacc.sh
```

---

### SD3.5 MCTS ablation (prompt / CFG / reward correction)

```bash
bash ~/SiD-DiT_tts_dev/run_mcts_ablation.sh
```

Ablation configs: `none prompt cfg correction prompt_cfg prompt_corr cfg_corr full`

```bash
# Quick test (5 sims, subset of ablations)
N_SIMS=5 ABLATIONS="none full" bash ~/SiD-DiT_tts_dev/run_mcts_ablation.sh

# Skip ablations that need Qwen (useful if model not cached yet)
ABLATIONS="none cfg correction cfg_corr" PRECOMPUTE_REWRITES=0 \
bash ~/SiD-DiT_tts_dev/run_mcts_ablation.sh
```

Outputs: `$SCRATCH/mcts_ablation/ablation_<timestamp>/`
Summary TSV: `$SCRATCH/mcts_ablation/ablation_<timestamp>/ablation_summary.tsv`

---

### SD3.5 dynamic CFG MCTS (node-adaptive CFG)

Runs `sd35_ddp_experiment_dynamic_cfg.py` via:
`run_sd35_dynamic_cfg_tacc.sh`

```bash
# Default: adaptive mode, cfg-only action space (no Qwen, no correction)
bash ~/SiD-DiT_tts_dev/run_sd35_dynamic_cfg_tacc.sh

# Fixed vs adaptive ablation
MCTS_CFG_MODES="fixed adaptive" \
N_SIMS=50 \
CFG_SCALES="1.0 1.25 1.5 1.75 2.0" \
bash ~/SiD-DiT_tts_dev/run_sd35_dynamic_cfg_tacc.sh
```

Useful overrides:
```bash
PROMPT_FILE=~/SiD-DiT_tts_dev/hpsv2_subset.txt
START_INDEX=0 END_INDEX=100
NUM_GPUS=8
MCTS_CFG_ROOT_BANK="1.0 1.5 2.0"
MCTS_CFG_ANCHORS="1.0 2.0"
MCTS_CFG_MIN_PARENT_VISITS=3
```

Outputs:
- Per mode: `$SCRATCH/sd35_dynamic_cfg/sd35_dynamic_cfg_<timestamp>/cfg_mode_<mode>/`
- Cross-mode summary: `cfg_mode_summary.tsv`

---

### Flux Schnell MCTS ablation (prompt / CFG)

```bash
bash ~/SiD-DiT_tts_dev/run_mcts_ablation_flux.sh
```

Ablation configs: `none prompt cfg prompt_cfg`

```bash
# Quick test
N_SIMS=5 ABLATIONS="none prompt_cfg" bash ~/SiD-DiT_tts_dev/run_mcts_ablation_flux.sh

# Skip Qwen precompute
ABLATIONS="none cfg" PRECOMPUTE_REWRITES=0 \
bash ~/SiD-DiT_tts_dev/run_mcts_ablation_flux.sh
```

Outputs: `$SCRATCH/mcts_ablation_flux/ablation_<timestamp>/`

---

## SLURM submission

```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 \
       --gres=gpu:a100:8 --time=24:00:00 \
       --partition=gpu \
       --wrap="bash ~/SiD-DiT_tts_dev/run_mcts_ablation_flux.sh"
```

Or for the full run:
```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 \
       --gres=gpu:a100:8 --time=24:00:00 \
       --partition=gpu \
       --wrap="bash ~/SiD-DiT_tts_dev/run_tacc.sh"
```

For dynamic CFG SD3.5:
```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 \
       --gres=gpu:a100:8 --time=24:00:00 \
       --partition=gpu \
       --wrap="MCTS_CFG_MODES='fixed adaptive' bash ~/SiD-DiT_tts_dev/run_sd35_dynamic_cfg_tacc.sh"
```

---

## Key env vars (all set automatically by tacc_setup.sh)

| Var | Value |
|---|---|
| `DATA_ROOT` | `$WORK/ls6` |
| `HF_HOME` | `$WORK/ls6/model_cache/huggingface` |
| `IMAGEREWARD_CACHE` | `$WORK/ls6/model_cache/ImageReward` |
| `HPS_ROOT` | `$WORK/ls6/model_cache/hpsv2` |
| `PYTHON_BIN` | `~/miniconda3/envs/sid_dit/bin/python` |
| `SKIP_INSTALL` | `1` (default — skip pip installs) |

Set `SKIP_INSTALL=0` only when rebuilding the environment.

---

## Disk layout

| Filesystem | Quota | Notes |
|---|---|---|
| `$HOME` (`/home1/...`) | ~20 GB | Do NOT store models here |
| `$WORK` (`/work/...`) | ~1 TB | Model caches — persistent |
| `$SCRATCH` (`/scratch/...`) | large | Job outputs — **purged after ~10 days inactive** |
