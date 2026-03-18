"""Legacy wrapper for unified sampler (ImageReward + MCTS prompt/cfg search)."""

import sys

from sampling_unified import main


if __name__ == "__main__":
    legacy_defaults = [
        "--search_method",
        "mcts",
        "--reward_type",
        "imagereward",
        "--n_variants",
        "3",
        "--n_sims",
        "50",
        "--ucb_c",
        "1.41",
        "--out_dir",
        "./mcts_prompt_out_cfg_augmented_search space",
        "--cfg_scales",
        "1.0",
        "1.25",
        "1.5",
        "1.75",
        "2.0",
        "2.25",
        "2.5",
        "2.75",
        "3.0",
        "3.25",
        "3.5",
        "3.75",
        "4.0",
        "4.25",
        "4.5",
        "4.75",
        "5.0",
    ]
    main(legacy_defaults + sys.argv[1:])
