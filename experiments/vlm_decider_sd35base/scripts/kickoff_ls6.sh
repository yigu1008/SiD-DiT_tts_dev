#!/usr/bin/env bash
# Wait for the env builds (setup_envs.sh logs) and then run the given stages on this node. Unattended; logs/kickoff.log.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; L=$RUN_DIR/setup/logs; STAGES=${STAGES:-smoke jevact ec}
log(){ echo "[$(date '+%F %T')] $*" | tee -a $RUN_DIR/logs/kickoff.log; }
ready(){ grep -q "setup_envs done" $L/setup_envs_sid_dit.log 2>/dev/null && grep -q "setup_envs done" $L/setup_envs_conda_chain.log 2>/dev/null; }
DEADLINE=$(( $(date +%s) + 3*3600 )); log "waiting for env builds (deadline 3 h)"
until ready; do [ "$(date +%s)" -ge "$DEADLINE" ] && { log "ENV BUILDS NOT DONE after 3 h; not starting"; exit 1; }; sleep 60; done
for m in REWARD_ENV_OK JEV_ENV_OK SID_DIT_ENV_OK; do grep -q $m $L/setup_envs_*.log || { log "ENV CHECK FAILED: $m missing; not starting"; exit 1; }; done
log "envs ready -> STAGES=[$STAGES]"; STAGES="$STAGES" bash $RUN_DIR/scripts/run_ls6.sh; log "run_ls6 exited rc=$?"
