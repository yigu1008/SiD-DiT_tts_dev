export HF_HOME="/data/ygu/.cache"
export PATH="/home/ygu/miniconda3/envs/sid_dit/bin:$PATH"

python geneval_mcts.py \
    --geneval_prompts /home/ygu/geneval/prompts/evaluation_metadata.jsonl \
    --geneval_python  /home/ygu/miniconda3/envs/geneval/bin/python \
    --geneval_repo    /home/ygu/geneval \
    --detector_path   /home/ygu/geneval/dectect/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
    --cfg_scales  \
    --steps 4 --n_sims 30 --n_samples 4 \
    --start_index 0 --end_index 10