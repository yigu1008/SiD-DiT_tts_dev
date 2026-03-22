#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/shell_env.sh"

# Run multiple experiment settings inside one job.
# Format:
#   MCTS_SWEEP="n_sims:ucb_c n_sims:ucb_c ..."
#   GA_SWEEP="ga_population:ga_generations:ga_mutation_prob ..."
MCTS_SWEEP="${MCTS_SWEEP:-40:1.2 80:1.41}"
GA_SWEEP="${GA_SWEEP:-24:12:0.10 48:20:0.15}"
OUT_ROOT_BASE="${OUT_ROOT:-/data/ygu/hpsv2_all_models_ddp_sweep}"

if [[ -z "${MCTS_SWEEP// }" ]]; then
  echo "Error: MCTS_SWEEP is empty." >&2
  exit 1
fi
if [[ -z "${GA_SWEEP// }" ]]; then
  echo "Error: GA_SWEEP is empty." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT_BASE}"

run_idx=0
for mcts_cfg in ${MCTS_SWEEP}; do
  IFS=':' read -r mcts_n_sims mcts_ucb_c <<< "${mcts_cfg}"
  if [[ -z "${mcts_n_sims:-}" || -z "${mcts_ucb_c:-}" ]]; then
    echo "Error: invalid MCTS_SWEEP token '${mcts_cfg}', expected n_sims:ucb_c." >&2
    exit 1
  fi

  for ga_cfg in ${GA_SWEEP}; do
    IFS=':' read -r ga_population ga_generations ga_mutation_prob <<< "${ga_cfg}"
    if [[ -z "${ga_population:-}" || -z "${ga_generations:-}" || -z "${ga_mutation_prob:-}" ]]; then
      echo "Error: invalid GA_SWEEP token '${ga_cfg}', expected pop:gens:mut_prob." >&2
      exit 1
    fi

    ucb_tag="${mcts_ucb_c//./p}"
    mut_tag="${ga_mutation_prob//./p}"
    combo_tag="$(printf 'exp%02d_mcts%s_ucb%s_ga%sx%s_mut%s' \
      "${run_idx}" "${mcts_n_sims}" "${ucb_tag}" "${ga_population}" "${ga_generations}" "${mut_tag}")"

    export N_SIMS="${mcts_n_sims}"
    export UCB_C="${mcts_ucb_c}"
    export GA_POPULATION="${ga_population}"
    export GA_GENERATIONS="${ga_generations}"
    export GA_MUTATION_PROB="${ga_mutation_prob}"
    export OUT_ROOT="${OUT_ROOT_BASE}/${combo_tag}"

    echo
    echo "======================================================================"
    echo "[sweep ${run_idx}] ${combo_tag}"
    echo "  N_SIMS=${N_SIMS} UCB_C=${UCB_C}"
    echo "  GA_POPULATION=${GA_POPULATION} GA_GENERATIONS=${GA_GENERATIONS} GA_MUTATION_PROB=${GA_MUTATION_PROB}"
    echo "  OUT_ROOT=${OUT_ROOT}"
    echo "======================================================================"

    bash "${SCRIPT_DIR}/hpsv2_all_models_ddp_suite.sh" "$@"
    run_idx=$((run_idx + 1))
  done
done

echo "Finished ${run_idx} in-job experiments. Root: ${OUT_ROOT_BASE}"
