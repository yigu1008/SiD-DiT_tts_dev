#!/usr/bin/env python3
"""OPTPROMPT.md: per-version tables for the deliberate prompt optimizer runs (probe/opt_prompt_v1, probe/opt_prompt (v2), probe/opt_prompt_v3 ...)."""
import json, os, glob, collections, statistics as st
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"
VERS = [("v1", f"{R}/probe/opt_prompt_v1", "lenient single-pass readout, strict guard (0.5), 512 px"),
        ("v2", f"{R}/probe/opt_prompt", "observe-then-grade readout + per-element Jev verification, strict guard (0.5), 512 px"),
        ("v3", f"{R}/probe/opt_prompt_v3", "v2 readout + guard allowing implied contrasts (thresh 0.7), no-repeat rule, 896 px previews"),
        ("v4", f"{R}/probe/opt_prompt_v4", "v3 + proposer sees statuses only, original-vocabulary rule, lexical retention AND no-new-content filter"),
        ("v5", f"{R}/probe/opt_prompt_v5", "v4 but lexical filter = retention only (Jev guards additions), sampled retries for empty proposals")]
L = []; P = L.append
P("# Deliberate discrete prompt optimizer (`opt_prompt`) — probe results\n")
P("Loop per prompt: checklist (VLM) -> plain 28-step sample (seed 42, CFG 4.5) -> per-element grade -> K=4 typed edits -> Jev faithfulness guard -> Jev choice "
  "(or keep) -> repeat (max 4 rounds). Output = round with most `present` elements (ties -> latest). No reward inside the loop; ImageReward shown for reference only.\n")
for tag, d, desc in VERS:
    f = f"{d}/records.jsonl"
    if not os.path.exists(f) or os.path.getsize(f) == 0: P(f"## {tag}: {desc}\n\n(no records yet)\n"); continue
    recs = [json.loads(l) for l in open(f)]
    P(f"## {tag}: {desc}\n"); P("| prompt | r0 present | out round | out present | stop | IR r0 -> out | rounds | VLM s | Jev calls |"); P("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        r0 = r["rounds"][0]; fb = [x for x in r["rounds"] if x["round"] == r["final_round"]][0]
        P(f'| {r["prompt_id"]} {r["prompt"][:48]}… | {r0["n_present"]}/{r0["n_elements"]} | {r["final_round"]} | {r["final_present"]} | {r["stop_reason"]} | {r0["ir"]:.2f} -> {fb["ir"]:.2f} | {len(r["rounds"])} | {r["vlm_seconds"]:.0f} | {r.get("n_jev_calls", 0)} |')
    edited = [r for r in recs if r["final_round"] > 0]; improved = [r for r in edited if int(r["final_present"].split("/")[0]) > r["rounds"][0]["n_present"]]
    gp = [c.get("guard_p_yes") for r in recs for x in r["rounds"] for c in x.get("candidates", []) if c.get("guard_p_yes") is not None]
    P(f"\n{len(recs)} prompts; {len(edited)} changed wording; {len(improved)} improved checklist completeness; mean wall {st.fmean(r['seconds'] for r in recs):.0f} s/prompt; "
      f"{len(gp)} guard calls, pass rate {sum(1 for g in gp if g >= 0.5)/max(1,len(gp)):.0%} at 0.5.\n")
    P("Final wordings that differ from the original:")
    for r in edited: P(f'- {r["prompt_id"]}: “{r["final_wording"]}”')
    P("")
# held-out judges on the hard-5 prompts: each version's output vs the round-0 original (opt_prompt_r0)
import csv
if os.path.exists(f"{R}/scores.csv") and os.path.exists(f"{R}/config/prompt_ids_hard5.txt"):
    ids = [l.strip() for l in open(f"{R}/config/prompt_ids_hard5.txt") if l.strip()]
    t = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{R}/scores.csv")):
        if r["prompt_id"] in ids: t[(r["arm"], r["prompt_id"])][r["metric"]] = float(r["score"])
    mets = ["ir", "hpsv3", "pickscore", "hpsv2", "vqascore", "qwen_alignment"]; arms = ["opt_prompt_r0", "opt_prompt_v3", "opt_prompt_v4", "opt_prompt_v5"]
    have = [a for a in arms if any((a, i) in t for i in ids)]
    if have:
        common = [i for i in ids if all((a, i) in t and all(m in t[(a, i)] for m in mets) for a in have)]
        P(f"## Held-out judges on the hard prompts (mean over the {len(common)} prompts judged in every arm: {', '.join(common)}; r0 = original wording, same seed)\n")
        P("| arm | " + " | ".join(mets) + " |"); P("|---|" + "---|" * len(mets))
        for a in have:
            if common: P(f"| {a} | " + " | ".join(f"{st.fmean(t[(a, i)][m] for i in common):.3f}" for m in mets) + " |")
        P("\nPer-prompt deltas vs r0 (hpsv3 / pickscore / vqascore / ir):\n")
        for a in have[1:]:
            for i in ids:
                if (a, i) in t and ("opt_prompt_r0", i) in t and all(m in t[(a, i)] and m in t[("opt_prompt_r0", i)] for m in ("hpsv3", "pickscore", "vqascore", "ir")):
                    d = lambda m: t[(a, i)][m] - t[("opt_prompt_r0", i)][m]
                    P(f"- {a} {i}: hpsv3 {d('hpsv3'):+.2f}, pickscore {d('pickscore'):+.2f}, vqascore {d('vqascore'):+.3f}, ir {d('ir'):+.2f}")
        P("")
# cohort n=25 section
cf = f"{R}/main/opt_prompt/records.jsonl"
if os.path.exists(cf) and os.path.getsize(cf) > 0:
    recs = [json.loads(l) for l in open(cf)]
    P(f"## Cohort run: first 25 prompts of the 99-cohort (v5 configuration), output vs round-0 original wording\n")
    ch = [r for r in recs if r["final_round"] > 0]; imp = [r for r in ch if int(r["final_present"].split("/")[0]) > r["rounds"][0]["n_present"]]
    P(f"{len(recs)} prompts; round-0 all-present {sum(1 for r in recs if r['rounds'][0]['n_present']==r['rounds'][0]['n_elements'])}; wording changed {len(ch)}; checklist completeness improved {len(imp)}; "
      f"mean present fraction {st.fmean(r['rounds'][0]['n_present']/r['rounds'][0]['n_elements'] for r in recs):.2f} -> {st.fmean(int(r['final_present'].split('/')[0])/r['rounds'][0]['n_elements'] for r in recs):.2f}; "
      f"stop reasons {dict(collections.Counter(r['stop_reason'] for r in recs))}; mean {st.fmean(r['seconds'] for r in recs):.0f} s/prompt.\n")
    P("| prompt | r0 | out | round | stop | final wording (if changed) |"); P("|---|---|---|---|---|---|")
    for r in recs:
        P(f'| {r["prompt_id"]} {r["prompt"][:40]}… | {r["rounds"][0]["n_present"]}/{r["rounds"][0]["n_elements"]} | {r["final_present"]} | {r["final_round"]} | {r["stop_reason"]} | {("“"+r["final_wording"][:80]+"”") if r["final_round"]>0 else ""} |')
    cfile = f"{R}/main/contrast_n25_opt_prompt_minus_r0.txt"
    if os.path.exists(cfile):
        txt = open(cfile).read().strip()
        P("\nPaired contrasts, opt_prompt minus original wording (same seed), 10,000 prompt-bootstrap resamples, 95% intervals; zero-crossing = inconclusive:\n")
        P("```"); P(txt); P("```")
    P("")
P("## Eye-check notes (2026-10-17)\n")
P("- v1 readout passed everything (5/5 prompts all-present at round 0) although the dresses image had three long hemlines and the birds had equal tails -> too lenient on comparison/differentiation elements; element texts lost their subjects (“1”, “left”).")
P("- v2 readout (observe first, then grade; Jev verifies each element on the image) caught both: dresses 2/6, birds 2/6 at round 0. On the birds the clarify edit “the higher-flying bird having a long tail” produced a long tail streamer on the higher bird (5/6; ImageReward 0.72 -> 1.06 without optimizing it). On the dresses every round still rendered four long gowns; the only edit that states the implied contrast was rejected by the strict guard (P=0.07).")
P("- Jev's per-element verification is conservative: it marked “exactly two birds” and “chasing” wrong on images that plainly show them, so `present` counts are deflated by ~1 element; consistent across rounds, so the output rule is unaffected.")
P("- The checklist occasionally adds an element the prompt does not state (“the scene is not in a closet”); it is graded missing every round and never drives an edit because the guard blocks added content.")
P("- Oscillation: with revert allowed, Jev alternated between the original and the improved wording (birds rounds 2-4); fixed by the no-repeat rule in v3.")
P("- Guard v2 test on hand-written candidates (live Jev): implied-contrast edit 0.98, explicit-count edit 0.99, paraphrase 1.00, contradiction 0.02, added content 0.00, dropped elements 0.58 -> threshold raised to 0.7.")
open(f"{R}/OPTPROMPT.md", "w").write("\n".join(L) + "\n"); print("OPTPROMPT.md written")
