#!/usr/bin/env python3
"""Is Jev's REWARD choice driven by the evidence, or is it a lexical constant?

The CFG-policy probe showed Jev answers control questions from a prior (withholding the image flipped
0/8 choices). The reward-selection question is a MATCHING question over text evidence, which should be
the kind it answers from evidence -- but that has to be measured, not assumed.

Each cohort prompt is put in the real reward_for() state template under five conditions:
  no_evidence            no VLM reports at all                -> reveals the lexical prior
  synth_content          reports describing missing/broken CONTENT      -> correct answer imagereward
  synth_appearance       reports describing bad APPEARANCE              -> correct answer hpsv3
  synth_kind             reports describing the wrong KIND of picture   -> correct answer pickscore
  synth_mixed_<x>        two classes named, one called the bigger problem -> correct answer follows that one
Accuracy over the three single-class conditions is the headline; chance is 1/3.
"""
from __future__ import annotations
import argparse, csv, json, sys, urllib.request
sys.path.insert(0, "/data/ygu/runs/vlm_decider_sd35base_2026-10-16/scripts")
from jev_reward_select import REWARD_FOCUS, SEARCH_REWARDS, post
PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"

SYNTH = {
 "synth_content": ("imagereward", [
   "The image shows the main subject, but one of the objects the prompt names is completely absent and another is duplicated. The hands of the figure are fused and the lettering on the sign is unreadable scribble.",
   "Most of the scene matches, but the two items the prompt places side by side are in the wrong order and one of them is missing entirely. One animal has five legs.",
   "The subject is recognisable, but the count is wrong - the prompt asks for three and only one is shown - and the second object the prompt names does not appear anywhere in the frame."]),
 "synth_appearance": ("hpsv3", [
   "Everything the prompt asks for is present and correctly arranged, but the colours are heavily oversaturated, the highlights are blown out and the surfaces look waxy and plastic.",
   "All described elements are there and in the right places, but the lighting is flat and inconsistent, the texture is mushy, and there is visible noise and compression artifacting across the image.",
   "The content is complete and faithful, but the image is garish - contrast is pushed far too hard, skin and fabric look like smooth plastic, and fine detail has been smeared away."]),
 "synth_kind": ("pickscore", [
   "The content is clean and complete, but it is drawn as a flat cartoon illustration when the prompt asks for a documentary photograph.",
   "Everything named is present and well rendered, but the subject is tiny and lost in a cluttered wide shot, so the picture does not read as the close portrait the prompt describes.",
   "The elements are all correct and the finish is clean, but the whole image is styled as a modern studio render when the prompt calls for a 19th-century painting."]),
}
MIXED = {   # two classes named, one explicitly called the bigger problem
 "synth_mixed_content_wins": ("imagereward",
   ["The colours are a little oversaturated, but the far bigger problem is that the second object the prompt names is missing entirely and the figure's hands are fused.",
    "There is mild oversaturation, though what really breaks the image is the count: the prompt asks for three and only one appears.",
    "Contrast is pushed slightly too hard; much more seriously, one of the two named objects is absent and the text is unreadable."]),
 "synth_mixed_appearance_wins": ("hpsv3",
   ["One small accessory the prompt mentions is missing, but the dominant problem by far is that the image is violently oversaturated with blown highlights and plastic-looking surfaces.",
    "A minor detail is absent; the overwhelming issue is the flat, muddy lighting and heavy compression artifacts across the whole frame.",
    "One named item is slightly misplaced, but what ruins the image is the garish colour and the waxy, smeared texture everywhere."]),
}

def build_state(prompt, reports, step, steps):
    if reports:
        ev = "\n".join(f'  - image from wording {chr(65+i)} at CFG ~{4.5+0.5*i:.1f}: {r}' for i, r in enumerate(reports))
        head = (f'This search has finished {len(reports)} images for this prompt. A vision model was asked, for each '
                f'finished image, what matches the prompt and what is missing, wrong or distorted. Its reports:\n' + ev + "\n\n")
    else:
        head = ("This search has not yet produced any finished images for this prompt, so there are no vision-model "
                "reports to go on.\n\n")
    return (f'ORIGINAL PROMPT (the only target): "{prompt}"\n' + head +
            f'The search is at step {step} of {steps} and must now pick which single quality to optimise for the '
            f'rest of this prompt. Each option below names one class of deficiency. Choose the class that the '
            f'reports above show is the LIMITING problem for this prompt - the one most worth fixing next. '
            f'Do not choose a class the reports give no evidence for.')

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="/data/ygu/runs/vlm_decider_sd35base_2026-10-16/config/prompt_ids_rsel9.txt")
    p.add_argument("--jev_server", default="http://127.0.0.1:5161")
    p.add_argument("--out", default="/data/ygu/runs/vlm_decider_sd35base_2026-10-16/main/reward_focus_probe.jsonl")
    p.add_argument("--step", type=int, default=12); p.add_argument("--steps", type=int, default=28)
    a = p.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    opts = [REWARD_FOCUS[b] for b in SEARCH_REWARDS]
    conds = [("no_evidence", None, [])] + [(k, v[0], v[1]) for k, v in SYNTH.items()] + [(k, v[0], v[1]) for k, v in MIXED.items()]
    rows = []
    with open(a.out, "w") as f:
        for pid in ids:
            prompt = pmap[pid]
            for name, correct, reports in conds:
                st = build_state(prompt, reports, a.step, a.steps)
                r = post(a.jev_server + "/ask", {"state": st, "question": "Which class of deficiency is the limiting problem for this prompt?",
                                                 "options": opts, "orders": 3, "seed": 7, "tag": f"probe/{pid}/{name}"})
                pick = SEARCH_REWARDS[int(r["argmax"])]
                row = {"prompt_id": pid, "condition": name, "correct": correct, "chosen": pick,
                       "hit": (correct is not None and pick == correct), "probs": r["probs"]}
                rows.append(row); f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{pid:6s} {name:30s} -> {pick:12s}" + (f" (want {correct})" if correct else "") +
                      ("  HIT" if row["hit"] else ("  miss" if correct else "")), flush=True)
    print("\n================ summary ================")
    pr = {}
    for r in rows: pr.setdefault(r["condition"], []).append(r)
    for c, rs in pr.items():
        dist = {}
        for r in rs: dist[r["chosen"]] = dist.get(r["chosen"], 0) + 1
        acc = f"  acc {sum(r['hit'] for r in rs)}/{len(rs)}" if rs[0]["correct"] else ""
        print(f"{c:30s} {dist}{acc}")
    single = [r for r in rows if r["condition"] in SYNTH]
    mixed = [r for r in rows if r["condition"] in MIXED]
    print(f"\nsingle-class accuracy {sum(r['hit'] for r in single)}/{len(single)} (chance {len(single)/3:.1f})")
    print(f"mixed-class accuracy  {sum(r['hit'] for r in mixed)}/{len(mixed)} (chance {len(mixed)/3:.1f})")
    ne = [r["chosen"] for r in rows if r["condition"] == "no_evidence"]
    print(f"no-evidence prior: {dict((x, ne.count(x)) for x in set(ne))}")
    flips = sum(1 for pid in set(r["prompt_id"] for r in rows)
                if len({r["chosen"] for r in rows if r["prompt_id"] == pid and r["condition"] in SYNTH}) > 1)
    print(f"prompts whose choice CHANGES across the three single-class conditions: {flips}/{len(ids)}")

if __name__ == "__main__":
    main()
