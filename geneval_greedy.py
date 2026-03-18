"""Legacy wrapper for unified sampler (GenEval + greedy cfg search)."""

import sys

from sampling_unified import main


if __name__ == "__main__":
    legacy_defaults = [
        "--search_method",
        "greedy",
        "--reward_type",
        "geneval",
        "--n_variants",
        "0",
        "--n_samples",
        "4",
        "--out_dir",
        "./greedy_geneval_out",
        "--cfg_scales",
        "1.0",
        "1.5",
        "2.0",
        "2.5",
        "3.0",
        "4.0",
    ]
    main(legacy_defaults + sys.argv[1:])
