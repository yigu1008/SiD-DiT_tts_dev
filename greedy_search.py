"""Legacy wrapper for unified sampler (ImageReward + greedy cfg search)."""

import sys

from sampling_unified import main


if __name__ == "__main__":
    legacy_defaults = [
        "--search_method",
        "greedy",
        "--reward_type",
        "imagereward",
        "--n_variants",
        "0",
        "--out_dir",
        "./greedy_out",
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
