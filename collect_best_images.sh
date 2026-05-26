#!/usr/bin/env bash
# Walk a run root, find every method's best_images/ folder, and flat-copy
# each PNG with a method+prompt-tagged filename so you can drop into slides.
#
# Usage:
#   bash collect_best_images.sh <RUN_ROOT> <OUT_DIR>
#
# Example:
#   bash collect_best_images.sh \
#     /mnt/data/v-yigu/all_in_one/flux-newcfg/composite/flux_schnell/seed42 \
#     figures/raw/flux_best_images

set -euo pipefail
RUN_ROOT="${1:?usage: collect_best_images.sh <RUN_ROOT> <OUT_DIR>}"
OUT_DIR="${2:?usage: collect_best_images.sh <RUN_ROOT> <OUT_DIR>}"
mkdir -p "${OUT_DIR}"

n=0
while IFS= read -r src; do
    # Path looks like .../<method>/run_*/best_images/prompt_NNNN.png
    method="$(basename "$(dirname "$(dirname "${src}")")")"
    fname="$(basename "${src}")"
    dst="${OUT_DIR}/${method}__${fname}"
    cp -f "${src}" "${dst}"
    n=$((n + 1))
done < <(find "${RUN_ROOT}" -path '*/best_images/*.png' -type f)

echo "Copied ${n} images to ${OUT_DIR}"
ls -la "${OUT_DIR}" | head -20
