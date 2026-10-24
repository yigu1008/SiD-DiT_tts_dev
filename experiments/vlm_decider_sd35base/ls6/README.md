# LS6 (TACC) deployment files — copies of what lives in `$SCRATCH/jev_decider/{env.sh,sync_from_repo.sh,setup/}`
`env.sh` is the single source of paths; `sync_from_repo.sh` copies `scripts/ config/ deps/` from this checkout into `$RUN_DIR`
and remaps the A6000 paths; `setup_envs.sh` builds `$SCRATCH/envs/{reward,jev,vqascore}` (miniforge) and completes `sid_dit`.
Launch with `scripts/run_ls6.sh` / `run_ls6.sbatch` (see DEPLOY.md §LS6). Env.sh is deliberately absolute: it documents the run.
