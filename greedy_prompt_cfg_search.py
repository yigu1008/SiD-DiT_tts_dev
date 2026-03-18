"""Legacy wrapper for unified sampler (ImageReward + greedy prompt/cfg search)."""

import sys

from sampling_unified import main


if __name__ == "__main__":
    legacy_defaults = [
        "--search_method",
        "greedy",
        "--reward_type",
        "imagereward",
        "--n_variants",
        "3",
        "--out_dir",
        "./greedy_prompt_out_cfg",
        "--cfg_scales",
        "1.0",
        "1.25",
        "1.5",
        "1.75",
        "2.0",
        "2.25",
        "2.5",
    ]
    main(legacy_defaults + sys.argv[1:])
