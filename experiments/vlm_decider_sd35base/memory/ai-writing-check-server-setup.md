---
name: ai-writing-check-server-setup
description: "Binoculars AI-writing detector deployed on the GPU server 2026-09-26; env path, model cache, GPU, disk pitfalls, and how to run/serve it"
metadata: 
  node_type: memory
  type: reference
  originSessionId: cb928cd6-a9aa-4c91-bce4-ed781fe2d2d9
  modified: 2026-09-26T03:20:59.906Z
---

Deployed 2026-09-26 for the user's `ai-writing-check` skill (Binoculars, Falcon-7B + Falcon-7B-instruct).

- **Checker:** `~/ai-writing-check/check.py` (copy of the skill's script; skill source unpacked at `/data/ygu/ai_writing_check/ai-writing-check/`). Report format: list of `{summary{thresholds likely_ai 0.854, borderline 0.902}, chunks[{idx,pages,words,score,verdict,text}]}`; lower score = more AI-like.
- **Env:** `/home/ygu/miniconda3/envs/binoculars/bin/python` (Python 3.11, torch 2.14.0+cu126, transformers 5.17, accelerate 1.15, pymupdf 1.28). Plain `pip install torch` gives a **CUDA-13 build that the driver (575.64 / CUDA 12.9) cannot run** — must install from `https://download.pytorch.org/whl/cu126`. transformers 5 prints a harmless `torch_dtype` deprecation warning.
- **Weights:** `~/.cache/huggingface/hub/models--tiiuae--falcon-7b{,-instruct}` (safetensors only, 14 GB each). Run with `HF_HOME=/home/ygu/.cache/huggingface HF_HUB_OFFLINE=1`. Both models load in ~13 s from NVMe; ~28 GB GPU.
- **Run:** `CUDA_VISIBLE_DEVICES=5 HF_HOME=/home/ygu/.cache/huggingface HF_HUB_OFFLINE=1 <env>/python ~/ai-writing-check/check.py paper.pdf --json --out report.json`. Service for the Mac panel: same env, `check.py --serve 8765` in tmux, reached via `ssh -N -L 8765:127.0.0.1:8765 gpu`; stop it when done (holds 28 GB).
- **Smoke test passed** on `~/ai-writing-check/smoke/test.txt` (2 chunks, both human-like, 13 s).

**First real run (2026-09-26, `actdiff_iclr2027.pdf`, 47 pp):** `check.py` drops everything from the References heading on, so the appendix (p.13-47) must be scored from a page-split copy (`pymupdf` insert_pdf) and its page numbers re-offset; references span p.10-12, so find the appendix start by heading text, not by assuming the page after References. Outputs in `~/ai-writing-check/mirror/`: `actdiff_report.json` (main, 39 chunks), `actdiff_appendix_report.json` (102 chunks, appendix-pdf page numbers), `actdiff_report_full.json` (both, original page numbers, panel-ready), `actdiff_iclr2027_marked.pdf` (highlights by verdict + margin notes + legend). Result: 0% likely-AI, one borderline chunk that is a table header. **Running a batch `check.py` while the `--serve` process holds GPU 5 OOMs** — use another idle GPU (GPU 3 worked) or the service's `/score`. Service: tmux session `binoculars`, port 8765, ~27 GB on GPU 5; stop with `tmux kill-session -t binoculars`.

**Disk pitfall:** the root partition (`/`, holds home and miniconda) was at 3 GB free after the 27 GB download; /data is ~17 GB free. On 2026-09-26 the user approved deleting `/tmp/sid_reward_cache`, `/tmp/ollama.tar.zst`, `~/.cache/dnnlib`, and `/data/ygu/reward_orthogonality/images` (phase-1 images; scores kept in `merged_scores.csv`). `/tmp/xeh94gxI` and `/tmp/BlVGmAuS` are ygu-owned dirs whose contents belong to other users and cannot be removed. `pgrep -f` on a path that appears in the calling shell's own command line self-matches — use a `ps ... | grep -v 'bash -c'` filter instead. See [[image-reward-env-pitfalls]].
