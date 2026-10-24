# LS6 Jev-decider run: the single source of paths. `source` this in every shell, launcher and sbatch script.
# Layout: $RUN_DIR/{scripts,config,deps} = remapped copy of the repo's experiments/vlm_decider_sd35base (sync_from_repo.sh),
#         $RUN_DIR/{main,logs,state,scores.csv} = run outputs, $RUN_DIR/setup = env/weight builders, $RUN_DIR/slurm = sbatch.
S=/scratch/09398/yigu
export RUN_DIR=$S/jev_decider
export REPO_DIR=/home1/09398/yigu/SiD-DiT_tts_dev-jev
export HF_HOME=$S/hf_cache HF_HUB_CACHE=$S/hf_cache/hub HF_HUB_DISABLE_XET=1     # SD3.5/HPSv2/CLIP-H are symlinks into $WORK/model_cache
export IMAGEREWARD_CACHE=$S/model_cache/ImageReward HPS_ROOT=$S/model_cache/hpsv2 # -> $WORK/model_cache/{ImageReward,hpsv2}
export CONDA_BASE=$S/miniforge3
export SIDDIT_PY_PREFIX=$S/envs/sid_dit REWARD_PY_PREFIX=$S/envs/reward JEV_PY_PREFIX=$S/envs/jev VQA_PY_PREFIX=$S/envs/vqascore
export PIP_CACHE_DIR=$S/pip_cache XDG_CACHE_HOME=$S/xdg_cache TMPDIR=$S/tmp         # $HOME quota is full: nothing may cache there
export SID_FORCE_WANDB_STUB=1 WANDB_DISABLED=true HPSV3_SCORE_STYLE=raw
export PYTHONNOUSERSITE=1; unset PYTHONPATH   # TACC python3/3.9.7 module sets it; its 3.9 site-packages would shadow every env
